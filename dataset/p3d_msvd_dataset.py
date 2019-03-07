import numpy as np
import pandas as pd
import os
import torch

from os import listdir
from os.path import join, isfile
from .base_msvd_vectorset import BaseMSVDvectorset

FRAMES_PER_FILE = 16

class P3DMSVDvectorset(BaseMSVDvectorset):
	def __init__(self, directory='vector/msvd', max_frames=96, split='train'):

		super().__init__(directory, max_frames, split)
		self.path = join(directory, 'p3d_pre')

	def __getitem__(self, index):
		video = self.videos[index]
		target = self.targets[index].split()

		vid_path = join(self.path, video)
		vectors = get_vectors(vid_path, self.max_frames)

		return vectors, target

	def get_vectors(vid_path, max_frames):
		vector_files = [f for f in listdir(vid_path) if isfile(join(vid_path, f)) and '.npy' in f]
		vector_files = sorted(vector_files, key=lambda x: int(x.split('.')[0]))


		if FRAMES_PER_FILE * len(vector_files) > max_frames:
			slice_index = np.random.choice(len(vector_files) - int(max_frames / FRAMES_PER_FILE) + 1)
			vector_files = vector_files[slice_index:slice_index+int(max_frames / FRAMES_PER_FILE)]

		vector = np.concatenate([np.load(join(vid_path, f)).tolist() for f in vector_files])
		vector = [np.squeeze(elem, axis=0).tolist() for elem in np.vsplit(vector, vector.shape[0])]

		return vector