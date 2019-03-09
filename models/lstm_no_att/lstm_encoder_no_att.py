import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from lstm_highway_no_att import LSTMHighway 



class LSTMEncoder(nn.Module):
	def __init__(self, cnn_feature_size, lstm_input_size, hidden_size_encoder, hidden_size_decoder):
		super(LSTMEncoder, self).__init__()
		self.cnn_feature_size = cnn_feature_size
		self.lstm_input_size = lstm_input_size
		self.hidden_size_encoder = hidden_size_encoder
		self.hidden_size_decoder = hidden_size_decoder

		self.highway_transform = LSTMHighway(cnn_feature_size, lstm_input_size)
		self.h_projection = nn.Linear(2 * hidden_size_encoder, hidden_size_decoder)
		self.c_projection = nn.Linear(2 * hidden_size_encoder, hidden_size_decoder)
		self.two_layer_lstm = nn.LSTM(lstm_input_size, hidden_size_encoder, 2)
        

	def forward(self, vids_padded, vids_actual_lengths):
		"""
		@param vids_padded: (max_vid_length, batch_size, cnn_feature_size)
		@param vids_actual_lengths (List[int]): List of actual lengths for each of the source videos in the batch
		"""

		lstm_input = self.highway_transform(vids_padded)
		lstm_input = nn.utils.rnn.pack_padded_sequence(lstm_input, vids_actual_lengths)
		enc_hiddens, (h_n, c_n) = self.two_layer_lstm(lstm_input)
		enc_hiddens, _ = nn.utils.rnn.pad_packed_sequence(enc_hiddens)
		enc_hiddens = enc_hiddens.permute(1, 0, 2) # shape (batch_size, max_vid_length, hidden_size_encoder)
		h_n = torch.cat((h_n[0], h_n[1]), 1)
		c_n = torch.cat((c_n[0], c_n[1]), 1)
		init_decoder_hidden = torch.unsqueeze(self.h_projection(h_n), 0)
		init_decoder_cell = torch.unsqueeze(self.c_projection(c_n), 0)

		init_decoder_hidden = torch.cat((init_decoder_hidden, init_decoder_hidden), 0)
		init_decoder_cell = torch.cat((init_decoder_cell, init_decoder_cell), 0)		
		dec_init_state = (init_decoder_hidden, init_decoder_cell)
		return enc_hiddens, dec_init_state


