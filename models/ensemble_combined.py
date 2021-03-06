import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import json

from collections import namedtuple
from os.path import join
from torch.autograd import Variable

from Encoders.lstm_encoder import LSTMEncoder
from Encoders.p3d_encoder import P3DEncoder
from Decoders.ensemble_decoder import EnsembleDecoder
from typing import List, Tuple, Dict, Set, Union

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class EnsembleCombined(nn.Module):

    def __init__(self, file_path, cnn_feature_size=2048, lstm_input_size=1024, hidden_size_encoder=1024, hidden_size_decoder=1024, 
                    embed_size=512, att_projection_dim=1024, num_layers=2, device='cpu', dropout_rate=0.3, pre_embed=None):

        super().__init__()
        self.cnn_feature_size = cnn_feature_size
        self.lstm_input_size = lstm_input_size
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        self.embed_size = embed_size
        self.att_projection_dim = att_projection_dim
        self.num_layers = num_layers
        self.frame_pad_token = [0] * cnn_feature_size
        self.file_path = file_path
        self.vocab = json.load(open(self.file_path, 'r'))
        self.vocab_id2word = {v: k for k, v in self.vocab.items()}
        self.device = device
        self.dropout_rate = dropout_rate
        self.pre_embed = pre_embed
        #self.dropout = nn.Dropout(0.5)

        self.to_embeddings = nn.Embedding(len(self.vocab), embed_size, self.vocab['<pad>'])
        if pre_embed is not None:
            embed_tensor = torch.from_numpy(np.load(pre_embed)).float()
            self.to_embeddings.weight = nn.Parameter(embed_tensor)
            self.embed_size = embed_size = embed_tensor.size(1)
            #self.embed_projection = nn.Linear(embed_tensor.size(1), self.embed_size)

        self.lstm_encoder = LSTMEncoder(cnn_feature_size, lstm_input_size, hidden_size_encoder, hidden_size_decoder, num_layers, dropout_rate)
        self.p3d_encoder = P3DEncoder(cnn_feature_size, hidden_size_encoder, hidden_size_decoder, num_layers, dropout_rate) 
        self.decoder = EnsembleDecoder(embed_size, hidden_size_encoder, hidden_size_decoder, att_projection_dim, num_layers, device, dropout_rate)
        self.dec_init_hidden_tranform = nn.Linear(2 * hidden_size_decoder, hidden_size_decoder)
        self.dec_init_cell_tranform = nn.Linear(2 * hidden_size_decoder, hidden_size_decoder)
        self.target_vocab_projection = nn.Linear(hidden_size_decoder, len(self.vocab))
        # if encoder == 'p3d':
        #     self.encoder = P3DEncoder(cnn_feature_size, hidden_size_encoder , hidden_size_decoder, num_layers, dropout_rate) 
        #     self.decoder = LSTMDecoder(embed_size, hidden_size_encoder, hidden_size_decoder, att_projection_dim, num_layers, device, dropout_rate)
        #     self.target_vocab_projection = nn.Linear(hidden_size_decoder, len(self.vocab))
        # else:
        #     self.encoder = LSTMEncoder(cnn_feature_size, lstm_input_size, hidden_size_encoder, hidden_size_decoder, num_layers, dropout_rate)
        #     self.decoder = LSTMDecoder(embed_size, hidden_size_encoder, hidden_size_decoder, att_projection_dim, num_layers, device, dropout_rate)
        #     self.target_vocab_projection = nn.Linear(hidden_size_decoder, len(self.vocab))



    def forward(self, source, captions):
        """ 
        @param source: tuple consisting of data for lstm encoder (source[0]) and p3d encoder (source[1])
        """
        vids_actual_lengths, vids_padded = self.pad_vid_frames(source[0]) # Both sorted by actual vid length (decsending order)
        features_actual_lengths, features_padded = self.pad_vid_frames(source[1])

        captions_actual_lengths, captions_padded = self.pad_captions(captions) # captions_padded: (max_sent_length, batch_size)
        captions_padded_exclude_last = captions_padded[:-1]
        captions_padded_embedded = self.to_embeddings(captions_padded_exclude_last)  # (max_sent_length - 1, batch_size, embed_size)
        """if self.pre_embed is not None:
            captions_padded_embedded = self.embed_projection(captions_padded_embedded)"""

        enc_hiddens, lstm_dec_init_state_1, lstm_dec_init_state_2 = self.lstm_encoder(vids_padded, vids_actual_lengths)
        enc_masks = self.generate_masks(enc_hiddens, vids_actual_lengths)
        mapped_features, p3d_dec_init_state_1, p3d_dec_init_state_2 = self.p3d_encoder(features_padded, features_actual_lengths)
        p3d_masks = self.generate_masks(mapped_features, features_actual_lengths)

        dec_init_state_1 = (self.dec_init_hidden_tranform(torch.cat((lstm_dec_init_state_1[0], p3d_dec_init_state_1[0]), 1)), 
            self.dec_init_cell_tranform(torch.cat((lstm_dec_init_state_1[1], p3d_dec_init_state_1[1]), 1)))
       #if self.num_layers == 1:
        dec_init_state_2 = None
        """else:
            dec_init_state_2 = (self.dec_init_hidden_tranform(torch.cat((lstm_dec_init_state_2[0], p3d_dec_init_state_2[0]), 1)), 
                self.dec_init_cell_tranform(torch.cat((lstm_dec_init_state_2[1], p3d_dec_init_state_2[1]), 1)))"""
        
        outputs = self.decoder(enc_hiddens, enc_masks, mapped_features, p3d_masks, dec_init_state_1, dec_init_state_2, captions_padded_embedded)

        P = F.log_softmax(self.target_vocab_projection(outputs), dim=-1)
        target_masks = (captions_padded != self.vocab['<pad>']).float() # Zero out probabilities for which we have nothing in the captions
        target_words_log_prob = torch.gather(P, index=captions_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        prob = Variable(target_words_log_prob.data.exp())
        scores = target_words_log_prob.sum(dim=0)
        return scores, prob, target_words_log_prob


    def generate_masks(self, enc_hiddens, source_lengths) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (batch_size, max_source_length, hidden_size_encoder)
        @param source_lengths (List[int]): List of actual lengths for each of the elements in the batch.
        
        @returns enc_masks (Tensor): Tensor of masks of shape (batch_size, max_source_length)
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)


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
            sent_padded = [self.vocab['<start>']] + [self.vocab.get(word, self.vocab['<unk>']) for word in sent] + [self.vocab['<end>']] + [self.vocab['<pad>']] * (max_len - len(sent))
            captions_padded.append(sent_padded)
        captions_padded_tensor = torch.tensor(captions_padded, dtype=torch.long, device=self.device).permute(1, 0) # shape: (max_sent_length, batch_size)

        return captions_actual_lengths, captions_padded_tensor


    def beam_search(self, video, beam_size=10, max_decoding_time_step=70) -> List[Hypothesis]:
        """ Given a single video, perform beam search, yielding captions.
        @param video tuple(List[List[float]], List[List[float]]): a single video (consisting of frame vectors)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded caption, represented as a list of words
                score: float: the log-likelihood of the caption
        """
        video_length_vec, video_tensor = self.pad_vid_frames([video[0]])
        feature_length_vec, feature_tensor = self.pad_vid_frames([video[1]])
        

        lstm_src_encodings, lstm_dec_init_vec_1, lstm_dec_init_vec_2  = self.lstm_encoder(video_tensor, video_length_vec)
        lstm_src_encodings_att_linear = self.decoder.lstm_att_projection(lstm_src_encodings)

        p3d_src_encodings, p3d_dec_init_vec_1, p3d_dec_init_vec_2  = self.p3d_encoder(feature_tensor, feature_length_vec)
        p3d_src_encodings_att_linear = self.decoder.p3d_att_projection(p3d_src_encodings)

        dec_init_state_1 = (self.dec_init_hidden_tranform(torch.cat((lstm_dec_init_vec_1[0], p3d_dec_init_vec_1[0]), 1)), 
            self.dec_init_cell_tranform(torch.cat((lstm_dec_init_vec_1[1], p3d_dec_init_vec_1[1]), 1)))
        #if self.num_layers == 1:
        dec_init_state_2 = None
        """else:
            dec_init_state_2 = (self.dec_init_hidden_tranform(torch.cat((lstm_dec_init_state_2[0], p3d_dec_init_state_2[0]), 1)), 
                self.dec_init_cell_tranform(torch.cat((lstm_dec_init_state_2[1], p3d_dec_init_state_2[1]), 1)))"""

        h_tm0 = dec_init_state_1
        h_tm1 = dec_init_state_2
        #att_tm1 = torch.zeros(1, self.hidden_size_decoder, device=self.device)
        lstm_h_prev_dec = torch.ones(1, self.hidden_size_decoder, device=self.device)/self.hidden_size_encoder
        p3d_h_prev_dec = torch.ones(1, self.hidden_size_decoder, device=self.device)/self.hidden_size_encoder

        eos_id = self.vocab['<end>']

        hypotheses = [['<start>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_lstm_src_encodings = lstm_src_encodings.expand(hyp_num,
                                                     lstm_src_encodings.size(1),
                                                     lstm_src_encodings.size(2))

            exp_lstm_src_encodings_att_linear = lstm_src_encodings_att_linear.expand(hyp_num,
                                                                           lstm_src_encodings_att_linear.size(1),
                                                                           lstm_src_encodings_att_linear.size(2))

            exp_p3d_src_encodings = p3d_src_encodings.expand(hyp_num,
                                                     p3d_src_encodings.size(1),
                                                     p3d_src_encodings.size(2))

            exp_p3d_src_encodings_att_linear = p3d_src_encodings_att_linear.expand(hyp_num,
                                                                           p3d_src_encodings_att_linear.size(1),
                                                                           p3d_src_encodings_att_linear.size(2))


            y_tm1 = torch.tensor([self.vocab[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.to_embeddings(y_tm1)

            
            dec_state_1, dec_state_2, output_t = self.decoder.step(y_t_embed, lstm_h_prev_dec, p3d_h_prev_dec, h_tm0, h_tm1, 
                exp_lstm_src_encodings, exp_lstm_src_encodings_att_linear, None, exp_p3d_src_encodings, exp_p3d_src_encodings_att_linear, None)         
            #(h_t0, cell_t0), (h_t, cell_t), att_t  = self.decoder.step(y_t_embed, att_tm1, h_tm0, h_tm1,
                                                      #exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)
            (h_t0, cell_t0) = dec_state_1
            if self.num_layers > 1:
                (h_t, cell_t) = dec_state_2
            log_p_t = F.log_softmax(self.target_vocab_projection(output_t), dim=-1)

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
                if hyp_word == '<end>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)

            h_tm0 = (h_t0[live_hyp_ids], cell_t0[live_hyp_ids])
            if self.num_layers == 1:
                lstm_h_prev_dec = h_t0[live_hyp_ids]
                p3d_h_prev_dec = h_t0[live_hyp_ids]
                h_tm1 = None
            else:
                lstm_h_prev_dec = h_t[live_hyp_ids]
                p3d_h_prev_dec = h_t[live_hyp_ids]
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    def save_arguments(self, outpath, key, args=None):
        arg_dict = {}
        arg_dict['file_path'] = self.file_path
        arg_dict['cnn_feature_size'] = self.cnn_feature_size
        arg_dict['lstm_input_size'] = self.lstm_input_size
        arg_dict['hidden_size_encoder'] = self.hidden_size_encoder
        arg_dict['hidden_size_decoder'] = self.hidden_size_decoder
        arg_dict['embed_size'] = self.embed_size
        arg_dict['dropout_rate'] = self.dropout_rate
        arg_dict['num_layers'] = self.num_layers
        arg_dict['att_projection_dim'] = self.att_projection_dim
        if self.pre_embed is not None:
            arg_dict['pre_embed'] = self.pre_embed
        
        with open(join(outpath, '{}.json'.format(key)), 'w') as f:
            json.dump(arg_dict, f)

        if args is not None:
            with open(join(outpath, '{}_args.json'.format(key)), 'w') as f:
                json.dump(args, f)














