import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import os
import torch

from os import listdir
from os.path import basename, exists, isdir, isfile, join, normpath


from models import ResNetFeatureExtractor, P3DFeatureExtractor
from utils import get_input_output_args, make_safe_path

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def extract_msvd_features():
	directory, out =  get_input_output_args()
	res = False if 'p3d' in out else True
	make_safe_path(out)

	img_dirs = [join(directory, d) for d in listdir(directory) if isdir(join(directory, d))]

	model = ResNetFeatureExtractor() if res else P3DFeatureExtractor()
	model.to(device)
	model.eval()
	for img_dir in img_dirs:
		name = basename(normpath(img_dir)).split('.')[0]
		print(name)
		imgs = [join(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f)) and '.jpg' in f]
		imgs = sorted(imgs, key=lambda x: int(basename(normpath(x)).split('.')[0]))
		img_ids = sorted([int(basename(normpath(img)).split('.')[0]) for img in imgs])
		model.preprocess(imgs, img_ids, out, name, device)
		break


def extract_mpii_features():
	path = '../data/mpii/jpgAllFrames'
	out = '../data/mpii/p3d_pre'

	res = False if 'p3d' in out else True
	make_safe_path(out)
	video_dirs = [join(path, d) for d in listdir(path) if isdir(join(path, d))]
	img_dirs = []

	model = ResNetFeatureExtractor() if res else P3DFeatureExtractor()
	model.to(device)
	model.eval()
	for directory in video_dirs:
		img_dirs += [join(directory, d) for d in listdir(directory) if isdir(join(directory, d))]

	for img_dir in img_dirs:
		name = basename(normpath(img_dir))
		print(name)
		imgs = [join(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f)) and '.jpg' in f]
		imgs = sorted(imgs, key=lambda x: int(basename(normpath(x)).split('.')[0]))
		img_ids = sorted([int(basename(normpath(img)).split('.')[0]) for img in imgs])
		num_frames = len(img_ids)
		sample_rate = int(num_frames / 16) if num_frames / 5 < 16 else 5
		imgs = imgs[::sample_rate]
		img_ids = img_ids[::sample_rate]
		model.preprocess(imgs, img_ids, out, name, device)
		

def main():
	#extract_msvd_features()
	extract_mpii_features()

if __name__ == '__main__':
	main()