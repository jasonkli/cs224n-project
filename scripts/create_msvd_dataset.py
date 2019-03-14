import json
import pandas as pd
import random
import string

from os import listdir
from os.path import join, isfile

random.seed(37)


path = '../data/msvd/video_corpus.csv'
vid_path = '../data/msvd/videos'


def process_sentence(sentence):
	return [''.join([c for c in s if c not in string.punctuation]).lower().strip() for s in sentence.split()]

def add_counts(words, word_counts):
	for word in words:
		if word in word_counts:
			word_counts[word] += 1
		else:
			word_counts[word] = 1

def get_counts(train, samples):
	word_counts = {}
	for vid in train:
		for sent in samples[vid]:
			words = sent.split()
			add_counts(words, word_counts)
	return word_counts

def get_splits(videos, train_size=1200, val_size=100, test_size=669):
	"""num_vids = len(videos)
	indices = range(num_vids)
	test_indices = random.sample(indices, test_size)
	remaining = list(set(indices) - set(test_indices))
	val_indices = random.sample(remaining, val_size)
	train_indices = list(set(remaining) - set(val_indices))
	train = [videos[i] for i in range(num_vids) if i in train_indices]
	val = [videos[i] for i in range(num_vids) if i in val_indices]
	test = [videos[i] for i in range(num_vids) if i in test_indices]"""
	train = videos[:train_size]
	val = videos[train_size:train_size+val_size]
	test = videos[train_size+val_size:]

	return train, val, test


def generate_csv(vid_list, samples, split='train', outpath='../data/msvd'):
	outfile = join(outpath, '{}.csv'.format(split))
	out_list = []
	for vid in vid_list:
		captions = list(samples[vid])
		if split == 'train' or split ==  'val':
			for cap in sorted(captions, key=lambda x: len(x), reverse=True)[5:10]:
				out_list.append([vid, cap])
		else:
			out_list.append([vid, ','.join(captions)])
		#out_list.append([vid, random.choice(samples[vid])])
	df = pd.DataFrame(out_list, columns=['VideoID', 'Target'])
	df.to_csv(outfile)

def create_json(words, outpath='../data/msvd'):
	out_dict = {}
	out_dict['<pad>'] = 0
	out_dict['<start>'] = 1
	out_dict['<end>'] = 2
	out_dict['<unk>'] = 3
	for index, word in enumerate(words):
		out_dict[word] = index + 4

	with open(join(outpath, 'msvd_vocab.json'), 'w') as f:
		json.dump(out_dict, f)


def main():

	valid =  set([f.split('.')[0] for f in listdir(vid_path) if isfile(join(vid_path, f)) and '.avi' in f])
	df = pd.read_csv(path)
	df = df.loc[df['Language'] == 'English']
	df = df[['VideoID', 'Start', 'End', 'Description']].dropna()

	rows = df.values.tolist()
	samples = {}
	for row in rows:
		video = '{}_{}_{}'.format(row[0], row[1], row[2])
		if video not in valid:
			continue
		if video not in samples:
			samples[video] = set()

		words = process_sentence(row[3])
		samples[video].add(' '.join(words))

	videos = list(samples.keys())
	train, val, test = get_splits(videos)

	word_counts = get_counts(train, samples)
	create_json(list(word_counts.keys()))
	generate_csv(train, samples, 'train')
	generate_csv(train, samples, 'train_eval')
	generate_csv(val, samples, 'val')
	generate_csv(val, samples, 'val_eval')
	generate_csv(test, samples, 'test')

if __name__ == '__main__':
	main()