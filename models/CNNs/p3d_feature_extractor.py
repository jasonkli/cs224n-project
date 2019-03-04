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
extractor = P3D199(pretrained=True)
size = 160

class P3DFeatureExtractor(BaseFeatureExtractor):

	def __init__(self):
		super().__init__(extractor, size)

	def extract_features(self, imgs, img_ids, out, name, device):
		avg_dir = join(out, 'avg')
		sample_dir = join(out, 'sample')
		avg_dir = join(avg_dir, name)
		sample_dir = join(sample_dir, name)
		make_clean_path(avg_dir)
		make_clean_path(sample_dir)

		possible_start_points = np.array(range(len(imgs))[:-15])
		if possible_start_points.shape[0] >= SAMPLE_SIZE:
			splits = np.array_split(possible_start_points, SAMPLE_SIZE)
			start_points = []
			for split in splits:
				start_points.append(np.random.choice(split))
			start_points = np.array(start_points)
		else:
			start_points = possible_start_points

		sequences = []
		for start in start_points:
			sequence = []
			for i in range(start, start+16):
				img = Image.open(imgs[i]).convert('RGB')
				img = transform_img(img, size=self.size)
				sequence.append(img)
			sequence = torch.stack(sequence, dim=1)
			sequences.append(sequence)

		x = torch.stack(sequences)
		with torch.no_grad():
			features = self(x.to(device))

		for i in range(features.size()[0]):
			feature = features[i].view(features[i].size()[0], -1).transpose(0, 1)
			torch.save(features, join(sample_dir, '{}.pt'.format(img_ids[start_points[i]])))

		avg_feature = torch.mean(features, dim=0)
		avg_feature = avg_feature.view(features[i].size()[0], -1).transpose(0, 1)
		torch.save(avg_feature, join(avg_dir, 'avg.pt'))
