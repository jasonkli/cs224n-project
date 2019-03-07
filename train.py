import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset import LSTMMSVDDataset
from models import LSTMCombined
from os.path import join
from utils import custom_collate_fn, make_clean_path

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_arguments():
	parser = argparse.ArgumentParser(description="Train arguements")
	parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Learning rate')
	parser.add_argument('--clip', dest='clip', type=float, default=5.0, help='Clip gradient')
	parser.add_argument('--max_epochs', dest='max_epochs', type=int, default=30, help="Max epochs")
	parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help="Batch size")
	parser.add_argument('--sgd', dest='sgd', action="store_true", help='Use sgd instead of adam')
	parser.add_argument('--train_iter', dest='train_iter', type=int, default=10)
	parser.add_argument('--val_iter', dest='val_iter', type=int, default=1000)
	parser.add_argument('--max_frames', dest='max_frames', type=int, default=96)
	parser.add_argument('--patience', dest='patience', type=int, default=4)
	parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.1)
	parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4)
	parser.add_argument('--directory', dest='directory', type=str, default=None)
	parser.add_argument('--model', dest='model', type=str, choices=['lstm'], default='lstm')
	parser.add_argument('--save_dir', dest='save_dir', default='checkpoints/')

	return parser.parse_args()

def evaluate_ppl(model, val_loader):
	model.eval()

	cum_loss = 0.
	cum_words = 0.

	with torch.no_grad():
		for data, target in val_loader:
			loss = -model.forward(data, target).sum()

			cum_loss += loss.item()
			num_words = sum([len(s[1:]) for s in target])  # omitting leading `<s>`
			cum_words += num_words

		ppl = np.exp(cum_loss / cum_words)

	return ppl

def plot(x, y, path, label):
	plt.plot(x, y)
	plt.xlabel('Iteration')
	plt.ylabel(label)
	plt.savefig(path)
	plt.cla()
	plt.clf()


def train(args):
	if args.model == 'lstm':
		train_dataset = (LSTMMSVDDataset(directory=args.directory, max_frames=args.max_frames) if args.directory 
					else LSTMMSVDDataset(max_frames=args.max_frames, split='train'))
		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  collate_fn=custom_collate_fn)

		val_dataset = (LSTMMSVDDataset(directory=args.directory, max_frames=args.max_frames) if args.directory 
					else LSTMMSVDDataset(max_frames=args.max_frames, split='val'))
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 4, shuffle=False, collate_fn=custom_collate_fn)

		model = LSTMCombined('data/msvd/msvd_vocab.json', device=device)	

	model.to(device)
	for param in model.parameters():
		if len(param.size()) > 1:
			nn.init.xavier_normal_(param.data)

	if args.sgd:
		optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
	else:
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay_rate, patience=args.patience, verbose=True)


	iteration = 0
	patience = 0
	cum_loss = report_loss = cum_tgt_words = report_tgt_words = cum_examples = report_examples = 0
	valid_ppls = []
	train_ppls = []
	train_losses = []
	x_axis = []
	best_ppl = float('inf')
	train_time = begin_time = time.time()

	path = join(args.save_dir, str(int(begin_time)))
	make_clean_path(path)
	save_path = join(path, '{}_checkpoint.pth'.format(int(begin_time)))
	model.save_arguments(path, str(int(begin_time)), vars(args))

	print('Starting training...')
	for epoch in range(args.max_epochs):
		for data, target in train_loader:
			iteration += 1
			model.train()
			optimizer.zero_grad()
			losses = -model.forward(data, target)
			batch_loss = losses.sum()
			loss = batch_loss / args.batch_size
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), args.clip)
			optimizer.step()

			batch_losses_val = batch_loss.item()
			report_loss += batch_losses_val
			cum_loss += batch_losses_val

			num_words = sum([len(s[1:]) for s in target])
			report_tgt_words += num_words
			cum_tgt_words += num_words
			report_examples += args.batch_size
			cum_examples += args.batch_size

			if iteration % args.train_iter == 0:
				print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
					  'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, iteration,
																						 report_loss / report_examples,
																						 math.exp(report_loss / report_tgt_words),
																						 cum_examples,
																						 report_tgt_words / (time.time() - train_time),
																						 time.time() - begin_time))

				train_time = time.time()
				report_loss = report_tgt_words = report_examples = 0.

			if iteration % args.val_iter == 0:
				print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, iteration,
																						 cum_loss / cum_examples,
																						 np.exp(cum_loss / cum_tgt_words),
																						 cum_examples))
				

				print('Starting validation ...')
				valid_ppl = evaluate_ppl(model, val_loader) 
				scheduler.step(valid_ppl)

				valid_ppls.append(valid_ppl)
				train_losses.append(cum_loss)
				train_ppls.append(np.exp(cum_loss / cum_tgt_words))
				x_axis.append(iteration)

				plot(x_axis, valid_ppls, join(path, 'valid_ppl.png'), 'Validation Perplexity')
				plot(x_axis, train_losses, join(path, 'train_loss.png'), 'Train Loss')
				plot(x_axis, train_ppls, join(path, 'train_ppl.png'), 'Train Perplexity')
				
				if valid_ppl < best_ppl:
					best_ppl = valid_ppl
					torch.save(model.state_dict(), save_path)

				print('Validation: iter %d, dev. ppl %f, best ppl %f' % (iteration, valid_ppl, best_ppl))
				cum_loss = cum_examples = cum_tgt_words = 0.

def main():
	args = get_arguments()
	train(args)


if __name__ == '__main__':
	main()