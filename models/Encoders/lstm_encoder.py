import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent))

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from layers.lstm_highway import LSTMHighway 



class LSTMEncoder(nn.Module):
	def __init__(self, cnn_feature_size, lstm_input_size, hidden_size_encoder, hidden_size_decoder):
		super(LSTMEncoder, self).__init__()
		self.cnn_feature_size = cnn_feature_size
		self.lstm_input_size = lstm_input_size
		self.hidden_size_encoder = hidden_size_encoder
		self.hidden_size_decoder = hidden_size_decoder

		self.highway_transform = LSTMHighway(cnn_feature_size, lstm_input_size)
		self.dropout = nn.Dropout(0.2)
		#self.highway_transform = nn.Linear(cnn_feature_size, lstm_input_size)
		self.linear_transform_enc_dec_h = nn.Linear(2 * hidden_size_encoder, hidden_size_decoder)
		self.linear_transform_enc_dec_c = nn.Linear(2 * hidden_size_encoder, hidden_size_decoder)
		self.two_layer_lstm = nn.LSTM(lstm_input_size, hidden_size_encoder, 2)
        

	def forward(self, vids_padded, vids_actual_lengths):
		"""
		@param vids_padded: (max_vid_length, batch_size, cnn_feature_size)
		@param vids_actual_lengths (List[int]): List of actual lengths for each of the source videos in the batch
		"""

		lstm_input = self.dropout(self.highway_transform(vids_padded))
		lstm_input = nn.utils.rnn.pack_padded_sequence(lstm_input, vids_actual_lengths)
		enc_hiddens, (h_n, c_n) = self.two_layer_lstm(lstm_input)
		enc_hiddens, _ = nn.utils.rnn.pad_packed_sequence(enc_hiddens)
		enc_hiddens = enc_hiddens.permute(1, 0, 2) # shape (batch_size, max_vid_length, hidden_size_encoder)
		h_n = torch.cat((h_n[0], h_n[1]), 1)
		c_n = torch.cat((c_n[0], c_n[1]), 1)
		h_n_layer1 = self.linear_transform_enc_dec_h(h_n)
		h_n_layer2 = self.linear_transform_enc_dec_h(h_n)
		c_n_layer1 = self.linear_transform_enc_dec_c(c_n)
		c_n_layer2 = self.linear_transform_enc_dec_c(c_n)
		dec_init_state_1 = (h_n_layer1, c_n_layer1)
		dec_init_state_2 = (h_n_layer2, c_n_layer2)

		return enc_hiddens, dec_init_state_1, dec_init_state_2


