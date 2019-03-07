import numpy as np
import pandas as pd
import os
import torch

from os import listdir
from os.path import join, isfile
from .base_msvd_dataset import BaseMSVDDataset

FRAMES_PER_FILE = 16

class P3DMSVDDataset(BaseMSVDDataset):
	def __init__(self, directory='data/msvd', max_frames=96, split='train'):

		super().__init__(directory, max_frames, split)
		self.path = join(directory, 'p3d_pre')

	def __getitem__(self, index):
		video = self.videos[index]
		target = self.targets[index].split()

		vid_path = join(self.path, video)
		data_files = [f for f in listdir(vid_path) if isfile(join(vid_path, f)) and '.npy' in f]
		data_files = sorted(data_files, key=lambda x: int(x.split('.')[0]))


		if FRAMES_PER_FILE * len(data_files) > self.max_frames:
			slice_index = np.random.choice(len(data_files) - int(max_frames / FRAMES_PER_FILE) + 1)
			data_files = data_files[slice_index:slice_index+int(max_frames / FRAMES_PER_FILE)]

		data = np.concatenate([np.load(join(vid_path, f)).tolist() for f in data_files])
		data = [np.squeeze(elem, axis=0).tolist() for elem in np.vsplit(data, data.shape[0])]

		return data, target