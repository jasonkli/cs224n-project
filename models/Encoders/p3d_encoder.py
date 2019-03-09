import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent))

import torch
import torch.nn as nn

from layers.lstm_highway import LSTMHighway 

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class P3DEncoder(nn.Module):

	def __init__(self, feature_size=2048, attention_size=None, hidden_decoder_size=512, dropout_rate=0.0):
		
		super().__init__()
		if attention_size is not None:
			self.feature_transform = nn.Linear(feature_size, attention_size)
		self.hidden_transform = nn.Linear(feature_size, hidden_decoder_size)
		self.cell_transform = nn.Linear(feature_size, hidden_decoder_size)
		self.dropout = nn.Dropout(dropout_rate)


	def forward(self, features_padded, features_actual_lengths):
		sum_features = torch.sum(features_padded, dim=0)
		norm = torch.unsqueeze(torch.tensor(features_actual_lengths, device=device, dtype=torch.float), dim=1)
		mean_features = sum_features / norm
		if attention_size is not None:
			mapped_features = self.feature_transform(features_padded)
		else:
			mapped_features = features_padded
		hidden_layer1 = self.dropout(self.hidden_transform(mean_features))
		hidden_layer2 = self.dropout(self.hidden_transform(mean_features))
		cell_layer1 = self.dropout(self.cell_transform(mean_features))
		cell_layer2 = self.dropout(self.cell_transform(mean_features))
		return mapped_features.permute(1,0,2).contiguous(), (hidden_layer1, cell_layer1), (hidden_layer2, cell_layer2)