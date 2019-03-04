import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import argparse
import csv
import cv2
import os

from os import listdir
from os.path import basename, exists, isfile, join, normpath

from utils import *

MIN_FRAMES = 16

def extract_images(video, outpath):
	frequency = 10
	vidcap = cv2.VideoCapture(video)
	length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
	if length < MIN_FRAMES * frequency:
		frequency = int(length / MIN_FRAMES)

	success, image = vidcap.read()
	count = 0
	num_output = 0
	while success:
		if count % frequency == 0:
			cv2.imwrite(join(outpath, '{}.jpg'.format(count)), image)
			num_output += 1
		success, image = vidcap.read()
		count += 1

	assert num_output >- 16

	return num_output


def main():
	directory, out = get_input_output_args()

	if not exists(out):
		os.mkdir(out)

	videos =  [join(directory, f) for f in listdir(directory) if isfile(join(directory, f)) and 'avi' in f]
	frame_count = {}
	frame_count_exact = {}
	csv_file = open(join(out, 'frames.csv'), 'w')
	writer = csv.writer(csv_file)

	for i, video in enumerate(videos):
		name = basename(normpath(video)).split('.')[0]
		print(i, name)
		outpath = join(out, name)
		make_clean_path(outpath)

		count = extract_images(video, outpath)

		writer.writerow([name, count])
		if count in frame_count:
			frame_count[count] += 1
		else:
			frame_count[count] = 1

	csv_file.close()

	for key in sorted(frame_count):
		print(key, frame_count[key])


if __name__ == '__main__':
	main()