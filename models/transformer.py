# Adapted from: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
import json
import math
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

from collections import namedtuple
from os.path import join
from typing import List, Tuple, Dict, Set, Union

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

CONST = 10000

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# num_heads * head_dim = model_size

class Transformer(nn.Module):
	def __init__(self, file_path, input_feature_size=2048, embed_size=128, model_size=512, num_heads=8, num_layers=6,
					max_len_vid=40, max_len_caption=30, hidden_size=2048, dropout_rate=0.2, pre_embed=None):
		super().__init__()
		self.frame_pad_token = [0] * input_feature_size
		self.file_path = file_path
		self.vocab = json.load(open(self.file_path, 'r'))
		self.vocab_id2word = {v: k for k, v in self.vocab.items()}
		self.max_len_vid = max_len_vid
		self.max_len_caption = max_len_caption
		self.input_feature_size = input_feature_size
		self.embed_size = embed_size
		self.model_size = model_size
		self.num_heads = num_heads
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.dropout_rate = dropout_rate


		self.dropout1 = nn.Dropout(0.0)
		self.dropout2 = nn.Dropout(0.0)
		self.dropout3 = nn.Dropout(dropout_rate)
		self.feature_projection = nn.Linear(input_feature_size, model_size)
		self.encoder = Encoder(model_size, num_heads, max_len_vid, 16, hidden_size, dropout_rate)

		self.to_embeddings = nn.Embedding(len(self.vocab), embed_size, self.vocab['<pad>'])
		if pre_embed is not None:
			embed_tensor = torch.from_numpy(np.load(pre_embed)).float()
			self.to_embeddings.weight = nn.Parameter(embed_tensor)
			self.embed_size = embed_size = embed_tensor.size(1)

		self.embed_projection = nn.Linear(embed_size, model_size)
		self.decoder = Decoder(model_size, num_heads, max_len_caption, 8, hidden_size, dropout_rate)
		self.target_vocab_projection = nn.Linear(model_size, len(self.vocab))

	def forward(self, vid, captions):
		vids_actual_lengths, vids_padded = self.pad_vid_frames(vid, self.max_len_vid) # (batch_size, max_length, input_feature_size)
		vid_mask = self.generate_masks(vids_padded, vids_actual_lengths) # (batch_size, 1, max_source_length)

		captions_actual_lengths, captions_padded = self.pad_captions(captions, self.max_len_caption) 
		captions_padded_exclude_last = captions_padded[:,:-1] #  (batch_size, max_sent_length - 1, embed_size)
		captions_padded_embedded = self.to_embeddings(captions_padded_exclude_last)
		cap_mask = self.generate_masks(captions_padded_embedded, captions_actual_lengths, dec=True) #(batch_size, max_source_length, max_source_length)

		vids_padded = self.feature_projection(self.dropout1(vids_padded)) #(batch_size, max_length, model_size)
		enc_out = self.encoder(vids_padded, vid_mask)
		captions_padded_embedded = self.embed_projection(self.dropout2(captions_padded_embedded)) #(batch_size, max_sent_length - 1, model_size)
		outputs = self.decoder(captions_padded_embedded, enc_out, vid_mask, cap_mask) #(batch_size, max_sent_length - 1, model_size)

		outputs = outputs.transpose(0, 1) #(max_sent_length - 1, batch_size, model_size)
		captions_padded = captions_padded.transpose(0,1)
		P = F.log_softmax(self.target_vocab_projection(self.dropout3(outputs)), dim=-1) #(max_sent_length - 1, batch_size, vocab_size)
		target_masks = (captions_padded != self.vocab['<pad>']).float() # Zero out probabilities for which we have nothing in the captions
		target_words_log_prob = torch.gather(P, index=captions_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:] #(max_sent_length - 1, batch_size)
		prob = Variable(target_words_log_prob.data.exp()).to(device)
		scores = target_words_log_prob.sum(dim=0)
		return scores, prob, target_words_log_prob

	def save_arguments(self, outpath, key, args=None):
		arg_dict = {}
		arg_dict['file_path'] = self.file_path
		arg_dict['input_feature_size'] = self.input_feature_size
		arg_dict['model_size'] = self.model_size
		arg_dict['num_heads'] = self.num_heads
		arg_dict['num_layers'] = self.num_layers
		arg_dict['hidden_size'] = self.hidden_size
		arg_dict['embed_size'] = self.embed_size
		arg_dict['dropout_rate'] = self.dropout_rate
		
		with open(join(outpath, '{}.json'.format(key)), 'w') as f:
			json.dump(arg_dict, f)

		if args is not None:
			with open(join(outpath, '{}_args.json'.format(key)), 'w') as f:
				json.dump(args, f)



	def beam_search(self, video, beam_size=10, max_decoding_time_step=10) -> List[Hypothesis]:
		vids_actual_lengths, vids_padded = self.pad_vid_frames([video], self.max_len_vid)
		vid_mask = self.generate_masks(vids_padded, vids_actual_lengths) 
		vids_padded = self.feature_projection(vids_padded)
		src_encodings = self.encoder(vids_padded, vid_mask)


		eos_id = self.vocab['<end>']

		hypotheses = [['<start>']]
		hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=device)
		completed_hypotheses = []

		t = 0
		while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
			t += 1
			hyp_num = len(hypotheses)

			exp_src_encodings = src_encodings.expand(hyp_num,
													 src_encodings.size(1),
													 src_encodings.size(2))

			y_tm1 = torch.tensor([[self.vocab[word] for word in hyp] for hyp in hypotheses], dtype=torch.long, device=device)
			y_t_embed = self.to_embeddings(y_tm1)
			y_t_embed= self.embed_projection(y_t_embed)

			cap_mask = torch.from_numpy(np.triu(np.ones((1, t, t)), k=1).astype(np.uint8))
			cap_mask = cap_mask.to(device)
			output_t = self.decoder(y_t_embed, exp_src_encodings, vid_mask, cap_mask)

			"""dec_state_1, dec_state_2, output_t = self.decoder.step(y_t_embed, h_prev_dec, h_tm0, h_tm1, 
				exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)"""    

			log_p_t = F.log_softmax(self.target_vocab_projection(output_t), dim=-1)
			log_p_t = log_p_t[:,t-1]
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
				#new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
				try:
					new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
				except:
					return [Hypothesis(value=['a'], score=1)]

				if hyp_word == '<end>':
					completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
														   score=cand_new_hyp_score))
				else:
					new_hypotheses.append(new_hyp_sent)
					live_hyp_ids.append(prev_hyp_id)
					new_hyp_scores.append(cand_new_hyp_score)

			if len(completed_hypotheses) == beam_size:
				break

			live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=device)

			hypotheses = new_hypotheses
			hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=device)

		if len(completed_hypotheses) == 0:
			completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
												   score=hyp_scores[0].item()))

		completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

		return completed_hypotheses


	def generate_masks(self, x, source_lengths, dec=False) -> torch.Tensor:
		# x: (batch_size, length, model_size)
		# enc_masks: (batch_size, 1/length, length)

		
		enc_masks = torch.zeros(x.size(0), x.size(1), dtype=torch.uint8)
		for e_id, src_len in enumerate(source_lengths):
			enc_masks[e_id, src_len:] = 1  # What to exclude

		enc_masks = enc_masks.unsqueeze(1) 

		if dec:
			max_len = x.size(1)
			future_mask = torch.from_numpy(np.triu(np.ones((1,max_len, max_len)), k=1).astype(np.uint8))
			enc_masks = enc_masks | future_mask # Want to combine masks

		return enc_masks.to(device)

	def pad_vid_frames(self, source, max_len):
		source = sorted(source, key=lambda vid: len(vid), reverse=True)
		vids_actual_lengths = [len(vid) for vid in source]

		vids_padded = []
		for vid in source:
			vid_padded = vid + [self.frame_pad_token] * (max_len - len(vid))
			vids_padded.append(vid_padded)
		vids_padded_tensor = torch.tensor(vids_padded, dtype=torch.float, device=device) # shape: (batch_size, max_len, cnn_feature_size)

		return vids_actual_lengths, vids_padded_tensor

	def pad_captions(self, captions, max_len):
		captions = sorted(captions, key=lambda sent: len(sent), reverse=True)
		captions_actual_lengths = [len(sent)+2 for sent in captions]  # plus 2 for <start> and <end> tokens to be added

		captions_padded = []
		max_len -= 2 # length doesn't include <start> and <end> tokens to be added
		for sent in captions:
			sent_padded = [self.vocab['<start>']] + [self.vocab.get(word, self.vocab['<unk>']) for word in sent] + [self.vocab['<end>']] + [self.vocab['<pad>']] * (max_len - len(sent))
			captions_padded.append(sent_padded)
		captions_padded_tensor = torch.tensor(captions_padded, dtype=torch.long, device=device)# shape: (batch_size, max_len)

		return captions_actual_lengths, captions_padded_tensor

class Encoder(nn.Module):
	def __init__(self, model_size, num_heads, max_len, num_layers, hidden_size=2048,dropout=0.1):

		super().__init__()
		self.positional_encoder = PositionalEncoder(model_size, max_len, dropout=0.5)
		self.layers = nn.ModuleList([EncoderLayer(model_size, num_heads, hidden_size, dropout)
										for _ in range(num_layers)])
		self.layer_norm = nn.LayerNorm(model_size)

	def forward(self, vids_padded, mask):
		out = self.positional_encoder(vids_padded)
		for layer in self.layers:
			out = layer(out, mask)
		out = self.layer_norm(out)
		return out

class Decoder(nn.Module):
	def __init__(self, model_size, num_heads, max_len, num_layers, hidden_size=2048, dropout=0.1):
		super().__init__()
		self.positional_encoder = PositionalEncoder(model_size, max_len, dropout=dropout)
		self.layers = nn.ModuleList([DecoderLayer(model_size, num_heads, hidden_size, dropout)
										for _ in range(num_layers)])
		self.layer_norm = nn.LayerNorm(model_size)

	def forward(self, captions_padded_embedded, enc_out, vid_mask, cap_mask):
		out = self.positional_encoder(captions_padded_embedded)
		for layer in self.layers:
			out = layer(out, enc_out, vid_mask, cap_mask)
		out = self.layer_norm(out)
		return out

class EncoderLayer(nn.Module):
	def __init__(self, model_size, num_heads, hidden_size=2048, dropout=0.1):
		super().__init__()
		self.self_attention = MultiHeadAttention(model_size, num_heads, dropout)
		self.feed_forward = PositionwiseFeedForward(model_size, hidden_size, dropout)

		self.layer_norm1 = nn.LayerNorm(model_size)
		self.layer_norm2 = nn.LayerNorm(model_size)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, x, mask):
		x_normed = self.layer_norm1(x)
		x = x + self.dropout1(self.self_attention(x_normed, x_normed, x_normed, mask))
		x_normed = self.layer_norm2(x)
		x = x + self.dropout2(self.feed_forward(x_normed))
		return x

		"""out = self.self_attention(x, x, x, mask)
		out = self.feed_forward(out)
		#out = self.dropout(out)
		#out = self.layer_norm(x + out)
		return out"""

class DecoderLayer(nn.Module):
	def __init__(self, model_size, num_heads, hidden_size=2048, dropout=0.1):
		super().__init__()

		self.self_attention = MultiHeadAttention(model_size, num_heads, dropout)
		self.enc_dec_attention = MultiHeadAttention(model_size, num_heads, dropout)
		self.feed_forward = PositionwiseFeedForward(model_size, hidden_size, dropout)

		self.layer_norm1 = nn.LayerNorm(model_size)
		self.layer_norm2 = nn.LayerNorm(model_size)
		self.layer_norm3 = nn.LayerNorm(model_size)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.dropout3 = nn.Dropout(dropout)

	def forward(self, x, enc_out, vid_mask, cap_mask):
		x_normed = self.layer_norm1(x)
		x = x + self.dropout1(self.self_attention(x_normed, x_normed, x_normed, cap_mask))
		x_normed = self.layer_norm2(x)
		x = x + self.dropout2(self.enc_dec_attention(x_normed, enc_out, enc_out, vid_mask))
		x_normed = self.layer_norm3(x)
		x = x + self.dropout3(self.feed_forward(x_normed))
		return x
		
		"""out = self.self_attention(x, x, x, cap_mask)
		out = self.enc_dec_attention(out, enc_out, enc_out, vid_mask)
		out = self.feed_forward(out)
		return out"""

class PositionwiseFeedForward(nn.Module):
	def __init__(self, model_size, hidden_size=2048, dropout=0.1):
		super().__init__()

		self.layer1 = nn.Linear(model_size, hidden_size)
		#self.dropout = nn.Dropout(dropout)
		self.layer2 = nn.Linear(hidden_size, model_size)
		#self.layer_norm = nn.LayerNorm(model_size)

	def forward(self, x):
		#Input/output: (batch_size, length, model_size)
		out = F.relu(self.layer1(x))
		out = self.layer2(out)
		return out

class MultiHeadAttention(nn.Module):
	def __init__(self, model_size, num_heads, dropout=0.1):
		super().__init__()

		self.model_size = model_size
		self.num_heads = num_heads
		self.head_dim = model_size // num_heads

		self.q_transform = nn.Linear(self.model_size, self.model_size)
		self.k_transform = nn.Linear(self.model_size, self.model_size)
		self.v_transform = nn.Linear(self.model_size, self.model_size)
		self.final_projection =  nn.Linear(self.model_size, self.model_size)
		#self.dropout = nn.Dropout(dropout)
		#self.layer_norm = nn.LayerNorm(model_size)

	def forward(self, querys, keys, values, mask):
		# Querys/keys/values: (batch_size, length, model_size)
		batch_size = querys.size(0)

		res = querys

		# Shape should be: (batch_size, num_heads, length, head_dim)
		querys = self.q_transform(querys).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
		keys = self.q_transform(keys).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
		values = self.q_transform(values).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)

		attention_scores = compute_attention(querys, keys, values, self.head_dim, mask)
		out = attention_scores.transpose(1,2).contiguous().view(batch_size, -1, self.model_size)
		out = self.final_projection(out)
		#out = self.dropout(out)
		#out = self.layer_norm(res + out)
		return out # Output should be same as input

class PositionalEncoder(nn.Module):
	def __init__(self, model_size, max_len, dropout=0.1):
		super().__init__()

		self.model_size = model_size
		self.dropout = nn.Dropout(dropout)
		positional_embed = torch.zeros(1, max_len, model_size)
		for i in range(max_len):
			for j in range(0, model_size, 2):
				positional_embed[0, i, j] = math.sin(i / (CONST ** (j / self.model_size)))
				positional_embed[0, i, j+1] = math.cos(i / (CONST ** (j / self.model_size)))

		self.register_buffer('positional_embed', positional_embed)

	
	def forward(self, x):
		# Input/output: (batch_size, length, model_size)

		x *= math.sqrt(self.model_size)
		length = x.size(1)
		x = x + Variable(self.positional_embed[:,:length], requires_grad=False).to(device)
		x = self.dropout(x)
		return x

def compute_attention(querys, keys, values, head_dim, mask=None):
	#querys, keys, values: (batch, num_head, len, head_dim)
	#enc_masks: (batch_size, 1/length, length)
	e_t = torch.matmul(querys, keys.transpose(-2,-1)) / np.sqrt(head_dim)
	if mask is not None:
		mask = mask.unsqueeze(1)
		e_t = e_t.data.masked_fill_(mask.byte(), -float('inf'))
	a_t = F.softmax(e_t, dim=-1)
	output = torch.matmul(a_t, values)
	return output


if __name__ == '__main__':
	videos = [[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]
	captions = [["bad", "jason"], ["helen", "is", "good", "helen", "is"]]
	model = Transformer(file_path="dummy_vocab.json", input_feature_size=5, embed_size=4, model_size=512, num_heads=8, num_layers=6,
					max_len_vid=20, max_len_caption=20, hidden_size=16, dropout=0.1)
	print(model.forward(videos, captions))



