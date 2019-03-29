import argparse
import json
import numpy as np
import pandas as pd
import os
import torch

from collections import namedtuple
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from os import listdir
from os.path import basename, join, isfile, normpath

from models import LSTMCombined, Transformer, EnsembleCombined
from dataset import LSTMMSVDDataset, P3DMSVDDataset, EnsembleMSVDDataset

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_arguments():
	parser = argparse.ArgumentParser(description="Test arguements")
	parser.add_argument('--path', dest='path', type=str, required=True, help='Path to checkpoint')
	parser.add_argument('--data', dest='data', type=str, default='data/msvd/test.csv', help='Path to test data')
	parser.add_argument('--vid_path', dest='vid_path', type=str, default='data/msvd/imgs_pre', help='Path to test vocab')
	parser.add_argument('--transformer', dest='transformer', action="store_true", help='Use transformer')
	parser.add_argument('--ensemble', dest='ensemble', action="store_true", help='Use ensmeble')

	return parser.parse_args()

def get_test_data(test_file):
	df = pd.read_csv(test_file)
	videos = df['VideoID'].tolist()
	targets = df['Target'].tolist()

	vid_to_references = {}
	for i, vid in enumerate(videos):
		if vid not in vid_to_references:
			vid_to_references[vid] = []
		for target in targets[i].split(','):
			vid_to_references[vid].append(target.split())

	videos = []
	references = []

	for vid in vid_to_references:
		videos.append(vid)
		references.append(vid_to_references[vid])

	return videos, references

def get_vectors(path, max_frames):
	vector_files = [f for f in listdir(path) if isfile(join(path, f)) and '.npy' in f]
	vector_files = sorted(vector_files, key=lambda x: int(x.split('.')[0]))

	if len(vector_files) > max_frames:
		slice_index = np.random.choice(len(vector_files) - max_frames + 1)
		vector_files = vector_files[slice_index:slice_index + max_frames]

	vectors = [np.load(join(path, f)).tolist() for f in vector_files]
	return vectors



def compute_corpus_level_bleu_score(references, hypotheses):
	""" Given decoding results and reference sentences, compute corpus-level BLEU score.
	@param references (List[List[str]]): a list of gold-standard reference target sentences
	@param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
	@returns bleu_score: corpus-level BLEU score
	"""

	bleu_score = corpus_bleu(references,
							 [hyp.value for hyp in hypotheses], smoothing_function=SmoothingFunction().method4)
	return bleu_score

def decode(model, videos, references, vid_path, outfile, beam_size, max_step, ensemble=False):
	""" Performs decoding on a test set, and save the best-scoring decoding results.
	If the target gold-standard sentences are given, the function also computes
	corpus-level BLEU score.
	"""


	hypotheses = beam_search(model, videos, vid_path, beam_size, max_step, ensemble=ensemble)

	top_hypotheses = [hyps[0] for hyps in hypotheses]
	bleu_score = compute_corpus_level_bleu_score(references, top_hypotheses)
	

	count = 0
	with open(outfile, 'w') as f:
		f.write('{}\n'.format(bleu_score))
		for refs, hyps in zip(references, hypotheses):
			count += 1
			top_hyp = hyps[0]
			hyp_sent = ' '.join(top_hyp.value)
			print(hyp_sent)
			hyp_sent = hyp_sent.replace('<end>', '')
			all_refs = ', '.join([' '.join(ref) for ref in refs])
			all_hyps = [' '.join(h.value) for h in hyps]
			#print(refs, all_hyps)
			try:
				f.write(hyp_sent + '/////////' + all_refs  + '\n')
			except:
				continue
	print('Corpus BLEU: {}'.format(bleu_score * 100))
	print(count)


def beam_search(model, videos, vid_path, beam_size=10, max_decoding_time_step=20, max_frames=600, ensemble=False):
	""" Run beam search to construct hypotheses for a list of src-language sentences.
	@param model (NMT): NMT Model
	@param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
	@param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
	@param max_decoding_time_step (int): maximum sentence length that Beam search can produce
	@returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
	"""

	hypotheses = []
	print(ensemble)
	with torch.no_grad():
		for i, vid in enumerate(videos):
			path = join(vid_path, vid)
			if ensemble:
				vectors = EnsembleMSVDDataset.get_vectors(join('data/msvd/imgs_pre', vid), join('data/msvd/p3d_pre', vid), max_frames)
			else:
				if 'p3d' in vid_path:
					vectors = P3DMSVDDataset.get_vectors(path, max_frames)
				else:
					vectors = LSTMMSVDDataset.get_vectors(path, max_frames)
			example_hyps = model.beam_search(vectors, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
			hypotheses.append(example_hyps)
			if i % 100 == 0:
				print(i)
	return hypotheses


def main():
	args = get_arguments()
	voc_path = 'data/msvd/new_word2id.json'
	embed_path = 'data/msvd/word_embed.npy'
	beam_size = 15
	max_step = 30

	if args.ensemble:
		model = EnsembleCombined(voc_path, device=device, pre_embed=embed_path)
	else:
		if 'p3d' in args.vid_path:
			#model = LSTMCombined(args.vocab, device=device, encoder='p3d', pre_embed='data/msvd/word_embed.npy')
			if not args.transformer:
				model = LSTMCombined(voc_path, device=device, pre_embed=embed_path, encoder='p3d')	
			else:
				model = Transformer(voc_path, pre_embed=embed_path)
		else:
			#model = LSTMCombined(args.vocab, device=device, pre_embed='data/msvd/word_embed.npy')
			if not args.transformer:
				model = LSTMCombined(voc_path, device=device, pre_embed=embed_path)	
			else:
				model = Transformer(voc_path, pre_embed=embed_path)


	model.load_state_dict(torch.load(args.path,  map_location='cpu'))
	model.to(device)
	model.eval()
	videos, references = get_test_data(args.data)
	name = basename(normpath(args.path)).split('_')[0]

	id = 'trainbest' if  '2' in basename(normpath(args.path)).split('_')[1] else 'valbest'
	s = 'test' if 'test' in args.data else 'train'
	outdir = join('results', name)
	if not os.path.exists(outdir):
		os.mkdir(outdir)
		with open(join(outdir, '{}_args.json'.format(name)), 'w') as f:
				json.dump(vars(args), f)

	outpath = join(outdir, '{}_beam={}_{}_{}_{}.txt'.format(name, beam_size, id, s, basename(normpath(args.data)).split('.')[0]))


	decode(model, videos, references, args.vid_path, outpath, beam_size, max_step, ensemble=args.ensemble)


if __name__ == '__main__':
	main()





