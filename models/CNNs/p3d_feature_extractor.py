import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import numpy as np
import torch

from os.path import join
from PIL import Image

from .base_feature_extractor import BaseFeatureExtractor
from .p3d import P3D199
from utils import transform_img, make_clean_path

SAMPLE_SIZE = 5
NUM_FRAMES = 16
extractor = P3D199(pretrained=True)
size = 160

class P3DFeatureExtractor(BaseFeatureExtractor):

	def __init__(self):
		super().__init__(extractor, size)

	def preprocess(self, imgs, img_ids, out, name, device):
		outpath = join(out, name)
		make_clean_path(outpath)

		seq_len = len(imgs)
		num_iter = int(seq_len / NUM_FRAMES)
		remaining = seq_len % NUM_FRAMES
		start = np.random.choice(remaining+1)

		sequences = []
		for i in range(num_iter):
			sequence = []
			for i in range(start + i * NUM_FRAMES, start + (i + 1) * NUM_FRAMES):
				img = Image.open(imgs[i]).convert('RGB')
				img = transform_img(img, size=self.size)
				sequence.append(img)
			sequence = torch.stack(sequence, dim=1)
			sequences.append(sequence)

		x = torch.stack(sequences)
		features = self.extract_features(x, device)
		for i in range(features.size()[0]):
			feature = features[i].view(features[i].size()[0], -1).transpose(0, 1)
			np.save(join(outpath, '{}.npy'.format(img_ids[start + i * NUM_FRAMES])), features.cpu().numpy())

