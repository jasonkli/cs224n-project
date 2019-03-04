import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
import torch.nn as nn

class BaseFeatureExtractor(nn.Module):

	def __init__(self, extractor, size):
		super().__init__()
		self.extractor = extractor
		self.size = size

		for param in self.extractor.parameters():
			param.requires_grad = False

	def forward(self, x):
		return self.extractor(x)

	def extract_features(self, x, device='cpu'):
		with torch.no_grad():
			features = self(x.to(device))
		return features

	def preprocess(self):
		raise NotImplementedError