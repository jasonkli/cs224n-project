import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import json

from collections import namedtuple
from typing import List, Tuple, Dict, Set, Union

from lstm_encoder_no_att import LSTMEncoder
from lstm_decoder_no_att import LSTMDecoder

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])



class LSTMCombined(nn.Module):
    def __init__(self, cnn_feature_size, lstm_input_size, hidden_size_encoder, hidden_size_decoder, embed_size, frame_pad_token, file_path, device, dropout_rate=0.2):
        super(LSTMCombined, self).__init__()
        self.cnn_feature_size = cnn_feature_size
        self.lstm_input_size = lstm_input_size
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        self.embed_size = embed_size
        self.frame_pad_token = frame_pad_token # should be a list of 0s, size = cnn_feature_size
        self.vocab = json.load(open(file_path, 'r'))
        self.device = device

        self.to_embeddings = nn.Embedding(len(self.vocab), embed_size, self.vocab['<pad>'])
        self.encoder = LSTMEncoder(cnn_feature_size, lstm_input_size, hidden_size_encoder, hidden_size_decoder)
        self.decoder = LSTMDecoder(embed_size, hidden_size_decoder, device)
        self.target_vocab_projection = nn.Linear(hidden_size_decoder, len(self.vocab))



    def forward(self, source, captions):
        vids_actual_lengths, vids_padded = self.pad_vid_frames(source) # Both sorted by actual vid length (decsending order)
        _, dec_init_state = self.encoder(vids_padded, vids_actual_lengths)

        captions_actual_lengths, captions_padded = self.pad_captions(captions) # captions_padded: (max_sent_length, batch_size)
        captions_padded_exclude_last = captions_padded[:-1]
        captions_padded_embedded = self.to_embeddings(captions_padded_exclude_last)  # (max_sent_length - 1, batch_size, embed_size)
        decoder_outputs = self.decoder(dec_init_state, captions_padded_embedded)
        P = F.log_softmax(self.target_vocab_projection(decoder_outputs), dim=-1)
        target_masks = (captions_padded != self.vocab['<pad>']).float() # Zero out probabilities for which we have nothing in the captions
        target_words_log_prob = torch.gather(P, index=captions_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_words_log_prob.sum(dim=0)
        return scores



    def pad_vid_frames(self, source):
        """ Sort source video list by length of video (longest to shortest) and convert list of videos (frames) 
        into tensor with necessary padding for shorter videos (i.e. videos with fewer frames). 

        @param source (List[List[List[float]]): list of videos (frames)
        @returns vids_actual_lengths (List[int]): List of actual lengths for each of the source videos in the batch
        @returns vids_padded_tensor: tensor of (max_vid_length, batch_size, cnn_feature_size)
        """
        source = sorted(source, key=lambda vid: len(vid), reverse=True)
        vids_actual_lengths = [len(vid) for vid in source]

        vids_padded = []
        max_len = 0
        for vid in source:
            if len(vid) > max_len:
                max_len = len(vid)
        for vid in source:
            vid_padded = vid + [self.frame_pad_token] * (max_len - len(vid))
            vids_padded.append(vid_padded)
        vids_padded_tensor = torch.tensor(vids_padded, dtype=torch.float, device=self.device).permute(1, 0, 2) # shape: (max_vid_length, batch_size, cnn_feature_size)

        return vids_actual_lengths, vids_padded_tensor


    def pad_captions(self, captions):
        captions = sorted(captions, key=lambda sent: len(sent), reverse=True)
        captions_actual_lengths = [len(sent)+2 for sent in captions]  # plus 2 for <start> and <end> tokens to be added

        captions_padded = []
        max_len = captions_actual_lengths[0] - 2 # length doesn't include <start> and <end> tokens to be added
        for sent in captions:
            sent_padded = [self.vocab['<start>']] + [self.vocab[word] for word in sent] + [self.vocab['<end>']] + [self.vocab['<pad>']] * (max_len - len(sent))
            captions_padded.append(sent_padded)
        captions_padded_tensor = torch.tensor(captions_padded, dtype=torch.long, device=self.device).permute(1, 0) # shape: (max_sent_length, batch_size)

        return captions_actual_lengths, captions_padded_tensor


    def beam_search(self, video, beam_size=5, max_decoding_time_step=70) -> List[Hypothesis]:
        """ Given a single video, perform beam search, yielding captions.
        @param video (List[List[float]]): a single video (consisting of frame vectors)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded caption, represented as a list of words
                score: float: the log-likelihood of the caption
        """
        video_length_vec, video_tensor = self.pad_vid_frames([video], self.device)

        src_encodings, dec_init_vec_1, dec_init_vec_2  = self.encoder(video_tensor, video_length_vec)
        src_encodings_att_linear = self.decoder.att_projection(src_encodings)

        h_tm0 = dec_init_vec_1
        h_tm1 = dec_init_vec_2
        att_tm1 = torch.zeros(1, self.hidden_size_decoder, device=self.device)

        eos_id = self.vocab['<end>']

        hypotheses = [['<start>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.to_embeddings(y_tm1)

            (h_t0, cell_t0), (h_t, cell_t), att_t  = self.decoder.step(y_t_embed, att_tm1, h_tm0, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab_id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            h_tm0 = (h_t0[live_hyp_ids], cell_t0[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses


























