import numpy as np
import os
import torch

from os import listdir
from os.path import join, isfile
from .base_msvd_dataset import BaseMSVDDataset

class LSTMMSVDDataset(BaseMSVDDataset):
	def __init__(self, directory='data/msvd', max_frames=96, split='train'):

		super().__init__(directory, max_frames, split)
		self.path = join(directory, 'imgs_pre')

	def __getitem__(self, index):
		video = self.videos[index]
		target = self.targets[index].split()

		vid_path = join(self.path, video)
		vector_files = [f for f in listdir(vid_path) if isfile(join(vid_path, f)) and '.pt' in f]
		vector_files = sorted(vector_files, key=lambda x: int(x.split('.')[0]))
		vectors = [torch.load(join(vid_path, f)) for f in vector_files]

		if len(vectors) > self.max_frames:
			slice_index = np.random.choice(len(vectors) - self.max_frames + 1)
			vectors = vectors[slice_index:slice_index+self.max_frames]

		return vectors, target





