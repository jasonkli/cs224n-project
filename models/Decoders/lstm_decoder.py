import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size_encoder, hidden_size_decoder, att_projection_dim, num_decoder_layers, device, dropout_rate=0.2):
        super(LSTMDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        self.att_projection_dim = att_projection_dim
        self.num_decoder_layers = num_decoder_layers
        self.device = device

        self.layer1_lstm_cell = nn.LSTMCell(embed_size + hidden_size_encoder, hidden_size_decoder)
        self.layer2_lstm_cell = nn.LSTMCell(hidden_size_decoder, hidden_size_decoder)
        self.att_projection = nn.Linear(hidden_size_encoder, att_projection_dim)
        self.dec_hidden_projection = nn.Linear(hidden_size_decoder, att_projection_dim)
        self.correct_att_projection_dim = nn.Linear(att_projection_dim, 1)
        self.linear_transform = nn.Linear(hidden_size_decoder, hidden_size_decoder)
        self.dropout = nn.Dropout(dropout_rate)


    

    def forward(self, enc_hiddens, enc_masks, dec_init_state_1, dec_init_state_2, captions_padded_embedded):
        """
        @param enc_hiddens (Tensor): (batch_size, max_vid_length, hidden_size_encoder)
        @param enc_masks (Tensor): video masks (batch_size, max_vid_length)
        @param captions_padded_embedded (Tensor): (max_sent_length, batch_size, embed_size)
        """

        batch_size = enc_hiddens.size(0)
        h_prev_dec = torch.ones(batch_size, self.hidden_size_decoder, device=self.device)/self.hidden_size_encoder
        outputs = []
        enc_hiddens_proj = self.att_projection(enc_hiddens) # (batch_size, max_vid_length, att_projection_dim)
        dec_state_layer1 = dec_init_state_1
        dec_state_layer2 = dec_init_state_2
        for elem in torch.split(captions_padded_embedded, 1): 
            elem = torch.squeeze(elem, 0)  # elem shape (batch_size, embed_size)
            dec_state_layer1, dec_state_layer2, output_t = self.step(elem, h_prev_dec, dec_state_layer1, 
                dec_state_layer2, enc_hiddens, enc_hiddens_proj, enc_masks)
            outputs.append(output_t)
            if self.num_decoder_layers == 1:
                h_prev_dec = dec_state_layer1[0]
            else:
                h_prev_dec = dec_state_layer2[0]
        outputs = torch.stack(outputs, 0)

        return outputs


    def step(self, elem, h_prev_dec, dec_state_layer1, dec_state_layer2, enc_hiddens, enc_hiddens_proj, enc_masks):
        h_prev_dec = self.dec_hidden_projection(h_prev_dec) # shape (batch_size, att_projection_dim)
        h_prev_dec = torch.unsqueeze(h_prev_dec, 1)
        added = torch.add(enc_hiddens_proj, 1, h_prev_dec) # shape (batch_size, max_vid_length, att_projection_dim)
        e_t = self.correct_att_projection_dim(torch.tanh(added)) 
        e_t = torch.squeeze(e_t, 2) # shape (batch_size, max_vid_length)
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))
        alpha_t = F.softmax(e_t, dim = 1)
        unsq_alpha_t = torch.unsqueeze(alpha_t, 1)
        a_t = torch.bmm(unsq_alpha_t, enc_hiddens) 
        a_t = torch.squeeze(a_t, 1) # shape (batch_size, hidden_size_encoder)

        lstm_input = torch.cat((elem, a_t), 1) # shape (batch_size, embed_size + hidden_size_encoder)
        h, c = self.layer1_lstm_cell(lstm_input, dec_state_layer1)
        if self.num_decoder_layers == 1:
            output_t = self.dropout(torch.tanh(self.linear_transform(h)))
            return (h, c), None, output_t
        else:
            h_out, c_out = self.layer2_lstm_cell(h, dec_state_layer2)
            output_t = self.dropout(torch.tanh(self.linear_transform(h)))
            return (h, c), (h_out, c_out), output_t
        
            
    



      





























