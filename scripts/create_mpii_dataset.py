import json
import pandas as pd
import random
import string

from os import listdir
from os.path import join, isfile

random.seed(21)

def generate_csv(split, data, name):
	outfile = join('../data/mpii', '{}.csv'.format(name))
	out_list = []
	for d in data:
		video = d[0].split('\\')[0]
		if video in split:
			out_list.append([d[0].split('\\')[1], ' '.join(d[1])])
	df = pd.DataFrame(out_list, columns=['VideoID', 'Target'])
	df.to_csv(outfile)

def get_counts(data, split):
	word_counts = {}
	for d in data:
		video = d[0].split('\\')[0]
		if video in split:
			words = d[1]
			for word in words:
				if word in word_counts:
					word_counts[word] += 1
				else:
					word_counts[word] = 1
	return word_counts

def get_splits(filename):
	splits = {
		'training': [],
		'validation': [],
		'test': []
	}
	with open(filename, 'r') as f:
		for line in f:
			(vid_id, split) = line.split()
			splits[split].append(vid_id)
	return splits

def process_sentence(sent):
	sent = ''.join([c for c in sent if c not in string.punctuation])
	return [word.lower().strip() for word in sent.split()]

def get_path(id):
	directory = '_'.join(id.split('_')[:-1])
	return join(directory, id)

def create_json(words, outpath='../data/mpii'):
	out_dict = {}
	out_dict['<pad>'] = 0
	out_dict['<start>'] = 1
	out_dict['<end>'] = 2
	out_dict['<unk>'] = 3
	for index, word in enumerate(words):
		out_dict[word] = index + 4

	with open(join(outpath, 'mpii_vocab.json'), 'w') as f:
		json.dump(out_dict, f)

def main():
	splits = get_splits('../data/mpii/splits.csv')
	path = '../data/mpii/annotations-someone.csv'
	data = []
	with open(path, 'r') as f:
		for line in f:
			(id, sent) = line.split('\t')
			vid_path = get_path(id)
			sent = process_sentence(sent)
			data.append((vid_path, sent))

	word_counts = get_counts(data, set(splits['training']))
	create_json(list(word_counts.keys()))
	generate_csv(set(splits['training']), data, 'train')
	generate_csv(set(splits['validation']), data, 'val')
	generate_csv(set(splits['test']), data, 'test')

	print(len(word_counts))

if __name__ == '__main__':
	main()