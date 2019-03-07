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
		vector_files = [f for f in listdir(vid_path) if isfile(join(vid_path, f)) and '.npy' in f]
		vector_files = sorted(vector_files, key=lambda x: int(x.split('.')[0]))
		
		if len(vector_files) > self.max_frames:
			slice_index = np.random.choice(len(vector_files) - self.max_frames + 1)
			vector_files = vector_files[slice_index:slice_index+self.max_frames]

		vectors = [np.load(join(vid_path, f)).tolist() for f in vector_files]

		return vectors, target






