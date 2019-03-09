import numpy as np
import pandas as pd
import os
import torch

from os import listdir
from os.path import join, isfile
from torch.utils.data import Dataset

class BaseMSVDDataset(Dataset):

	def __init__(self, directory='data/msvd', max_frames=96, split='train'):

		super().__init__()
		csv_file = join(directory, '{}.csv'.format(split))
		self.df = pd.read_csv(csv_file)
		self.videos = self.df['VideoID'].tolist()
		self.targets = self.df['Target'].tolist()
		self.max_frames = max_frames

	def __len__(self):
		return len(self.targets)

	def __getitem__(self, index):
		video = self.videos[index]
		target = np.random.choice(self.targets[index].split(',')).split()

		vid_path = join(self.path, video)
		vectors = self.get_vectors(vid_path, self.max_frames)
		
		return vectors, target

	@staticmethod
	def get_vectors(vid_path, max_frames):
		raise NotImplementedError
