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

	def get_vectors(vid_path, max_frames):
		vector_files = [f for f in listdir(vid_path) if isfile(join(vid_path, f)) and '.npy' in f]
		vector_files = sorted(vector_files, key=lambda x: int(x.split('.')[0]))


		if FRAMES_PER_FILE * len(vector_files) > max_frames:
			slice_index = np.random.choice(len(vector_files) - int(max_frames / FRAMES_PER_FILE) + 1)
			vector_files = vector_files[slice_index:slice_index+int(max_frames / FRAMES_PER_FILE)]

		vectors = np.concatenate([np.load(join(vid_path, f)) for f in vector_files], axis=0)
		vectors = [np.squeeze(elem, axis=0).tolist() for elem in np.vsplit(vectors, vectors.shape[0])]

		return vectors