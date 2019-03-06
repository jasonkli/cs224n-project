import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F


class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size_encoder, hidden_size_decoder, device, dropout_rate=0.2):
        super(LSTMDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size_encoder = hidden_size_encoder
        self.hidden_size_decoder = hidden_size_decoder
        self.device = device

        self.layer1_lstm = nn.LSTM(embed_size, hidden_size_decoder)
        self.layer2_lstm_cell = nn.LSTMCell(2 * hidden_size_decoder, hidden_size_decoder, bias=True)
        self.att_projection = nn.Linear(hidden_size_encoder, hidden_size_decoder)
        self.combined_output_projection = nn.Linear(hidden_size_encoder + hidden_size_decoder, hidden_size_decoder)
        self.dropout = nn.Dropout(dropout_rate)


    

    def forward(self, enc_hiddens, enc_masks, dec_init_state_1, dec_init_state_2, captions_padded_embedded, captions_actual_lengths):
        """
        @param enc_hiddens (Tensor): (batch_size, max_vid_length, hidden_size_encoder)
        @param enc_masks (Tensor): video masks (batch_size, max_vid_length)
        @param captions_padded_embedded (Tensor): (max_sent_length, batch_size, embed_size)
        """

        layer1_input = nn.utils.rnn.pack_padded_sequence(captions_padded_embedded, captions_actual_lengths)
        layer1_hiddens, (h_n, c_n) = self.layer1_lstm(layer1_input, dec_init_state_1) # layer1_hiddens: (max_sent_length, batch_size, hidden_size_decoder)
        layer1_hiddens, _ = nn.utils.rnn.pad_packed_sequence(layer1_hiddens)

        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size_decoder, device=self.device)
        combined_outputs = []
        enc_hiddens_proj = self.att_projection(enc_hiddens) # (batch_size, max_vid_length, hidden_size_decoder)
        dec_state = dec_init_state_2
        for elem in torch.split(layer1_hiddens, 1):
            elem = torch.squeeze(elem, 0)
            elem_bar = torch.cat((elem, o_prev), 1)
            dec_state, o_t = self.step(elem_bar, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t
        combined_outputs = torch.stack(combined_outputs, 0)

        return combined_outputs


    def step(self, elem_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks):
        dec_state = self.layer2_lstm_cell(elem_t, dec_state)
        dec_hidden = dec_state[0]
        unsq_dec_hidden = torch.unsqueeze(dec_hidden, 1)
        e_t = torch.bmm(unsq_dec_hidden, enc_hiddens_proj.permute(0, 2, 1))
        e_t = torch.squeeze(e_t, 1)
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))
            
        alpha_t = F.softmax(e_t, dim = 1)
        unsq_alpha_t = torch.unsqueeze(alpha_t, 1)
        a_t = torch.bmm(unsq_alpha_t, enc_hiddens)
        a_t = torch.squeeze(a_t, 1)
        U_t = torch.cat((a_t, dec_hidden), 1)
        V_t = self.combined_output_projection(U_t)
        combined_output = self.dropout(torch.tanh(V_t))

        return dec_state, combined_output




























