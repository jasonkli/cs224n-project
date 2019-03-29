import numpy as np
import pandas as pd
import os
import torch

from os import listdir
from os.path import join, isfile
from torch.utils.data import Dataset

from .lstm_msvd_dataset import LSTMMSVDDataset
from .p3d_msvd_dataset import P3DMSVDDataset
from .base_msvd_dataset import BaseMSVDDataset

class EnsembleMSVDDataset(BaseMSVDDataset):

	def __init__(self, directory='data/msvd', max_frames=96, split='train'):

		super().__init__(directory, max_frames, split)
		self.path2d = join(directory, 'imgs_pre')
		self.path3d = join(directory, 'p3d_pre')

	def __getitem__(self, index):
		video = self.videos[index]
		target = np.random.choice(self.targets[index].split(',')).split()

		vid_path2d = join(self.path2d, video)
		vid_path3d = join(self.path3d, video)
		vectors1 = LSTMMSVDDataset.get_vectors(vid_path2d, self.max_frames)
		vectors2 = P3DMSVDDataset.get_vectors(vid_path3d, self.max_frames)
		return (vectors1, vectors2), target

	@staticmethod
	def get_vectors(vid_path2d, vid_path3d, max_frames):
		vectors1 = LSTMMSVDDataset.get_vectors(vid_path2d, max_frames)
		vectors2 = P3DMSVDDataset.get_vectors(vid_path3d, max_frames)
		return (vectors1, vectors2)