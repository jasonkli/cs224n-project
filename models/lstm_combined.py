import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import json
from Encoders.lstm_encoder import LSTMEncoder
from Decoders.lstm_decoder import LSTMDecoder
from typing import List, Tuple, Dict, Set, Union



class LSTMCombined(nn.Module):
    def __init__(self, file_path, cnn_feature_size=2048, lstm_input_size=1024, hidden_size_encoder=512, hidden_size_decoder=512, embed_size=256,  device='cpu', dropout_rate=0.2):
        super(LSTMCombined, self).__init__()
        self.cnn_feature_size = cnn_feature_size
        self.lstm_input_size = lstm_input_size
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        self.embed_size = embed_size
        self.frame_pad_token = [0] * cnn_feature_size
        self.vocab = json.load(open(file_path, 'r'))
        self.device = device

        self.to_embeddings = nn.Embedding(len(self.vocab), embed_size, self.vocab['<pad>'])
        self.encoder = LSTMEncoder(cnn_feature_size, lstm_input_size, hidden_size_encoder, hidden_size_decoder)
        self.decoder = LSTMDecoder(embed_size, hidden_size_encoder, hidden_size_decoder, device)
        self.target_vocab_projection = nn.Linear(hidden_size_decoder, len(self.vocab))



    def forward(self, source, captions):
        vids_actual_lengths, vids_padded = self.pad_vid_frames(source) # Both sorted by actual vid length (decsending order)
        enc_hiddens, dec_init_state_1, dec_init_state_2 = self.encoder(vids_padded, vids_actual_lengths)
        dec_init_state_1 = (torch.unsqueeze(dec_init_state_1[0], 0), torch.unsqueeze(dec_init_state_1[1], 0))
        enc_masks = self.generate_masks(enc_hiddens, vids_actual_lengths)

        captions_actual_lengths, captions_padded = self.pad_captions(captions) # captions_padded: (max_sent_length, batch_size)
        captions_padded_exclude_last = captions_padded[:-1]
        captions_padded_embedded = self.to_embeddings(captions_padded_exclude_last)  # (max_sent_length - 1, batch_size, embed_size)
        combined_outputs = self.decoder(enc_hiddens, enc_masks, dec_init_state_1, dec_init_state_2, captions_padded_embedded, [length - 1 for length in captions_actual_lengths])
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)
        target_masks = (captions_padded != self.vocab['<pad>']).float() # Zero out probabilities for which we have nothing in the captions
        target_words_log_prob = torch.gather(P, index=captions_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_words_log_prob.sum(dim=0)
        return scores


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
        @returns vids_padded: tensor of (max_vid_length, batch_size, cnn_feature_size)
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















