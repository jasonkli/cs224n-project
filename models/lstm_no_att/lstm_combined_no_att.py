import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import json

from collections import namedtuple
from os.path import join
from typing import List, Tuple, Dict, Set, Union


from .lstm_encoder_no_att import LSTMEncoder
from .lstm_decoder_no_att import LSTMDecoder

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])



class LSTMBasic(nn.Module):
    def __init__(self, file_path, cnn_feature_size=2048, lstm_input_size=1024, hidden_size_encoder=512, hidden_size_decoder=512, embed_size=256,  device='cpu', dropout_rate=0.0):
        super(LSTMBasic, self).__init__()
        self.cnn_feature_size = cnn_feature_size
        self.lstm_input_size = lstm_input_size
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        self.embed_size = embed_size
        self.frame_pad_token = [0] * cnn_feature_size
        self.file_path = file_path
        self.vocab = json.load(open(self.file_path, 'r'))
        self.vocab_id2word = {v: k for k, v in self.vocab.items()}
        self.device = device
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

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
            sent_padded = [self.vocab['<start>']] +  [self.vocab.get(word, self.vocab['<unk>']) for word in sent] + [self.vocab['<end>']] + [self.vocab['<pad>']] * (max_len - len(sent))
            captions_padded.append(sent_padded)
        captions_padded_tensor = torch.tensor(captions_padded, dtype=torch.long, device=self.device).permute(1, 0) # shape: (max_sent_length, batch_size)

        return captions_actual_lengths, captions_padded_tensor


    def save_arguments(self, outpath, key, args=None):
        arg_dict = {}
        arg_dict['file_path'] = self.file_path
        arg_dict['cnn_feature_size'] = self.cnn_feature_size
        arg_dict['lstm_input_size'] = self.lstm_input_size
        arg_dict['hidden_size_encoder'] = self.hidden_size_encoder
        arg_dict['hidden_size_decoder'] = self.hidden_size_decoder
        arg_dict['embed_size'] = self.embed_size
        arg_dict['dropout_rate'] = self.dropout_rate
        with open(join(outpath, '{}.json'.format(key)), 'w') as f:
            json.dump(arg_dict, f)

        if args is not None:
            with open(join(outpath, '{}_args.json'.format(key)), 'w') as f:
                json.dump(args, f)






























