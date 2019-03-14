import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size_decoder, device, dropout_rate=0.2):
        super(LSTMDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size_decoder = hidden_size_decoder
        self.device = device

        self.two_layer_lstm = nn.LSTM(embed_size, hidden_size_decoder, 2)
        self.dropout = nn.Dropout(dropout_rate)


    

    def forward(self, dec_init_state, captions_padded_embedded):
        """
        @param enc_hiddens (Tensor): (batch_size, max_vid_length, hidden_size_encoder)
        @param enc_masks (Tensor): video masks (batch_size, max_vid_length)
        @param captions_padded_embedded (Tensor): (max_sent_length, batch_size, embed_size)
        """
        dec_hiddens, (h_n, c_n) = self.two_layer_lstm(captions_padded_embedded, dec_init_state)
    

        return dec_hiddens


    




























