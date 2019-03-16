import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset import LSTMMSVDDataset, P3DMSVDDataset
from models import LSTMCombined
from models.lstm_no_att import LSTMBasic
from os.path import join
from utils import custom_collate_fn, make_clean_path

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_arguments():
	parser = argparse.ArgumentParser(description="Train arguements")
	parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Learning rate')
	parser.add_argument('--clip', dest='clip', type=float, default=3.0, help='Clip gradient')
	parser.add_argument('--max_epochs', dest='max_epochs', type=int, default=3000, help="Max epochs")
	parser.add_argument('--batch_size', dest='batch_size', type=int, default=256, help="Batch size")
	parser.add_argument('--sgd', dest='sgd', action="store_true", help='Use sgd instead of adam')
	parser.add_argument('--train_iter', dest='train_iter', type=int, default=200)
	parser.add_argument('--val_iter', dest='val_iter', type=int, default=2000)
	parser.add_argument('--max_frames', dest='max_frames', type=int, default=32)
	parser.add_argument('--patience', dest='patience', type=int, default=64)
	parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.3)
	parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4)
	parser.add_argument('--gamma', dest='gamma', type=float, default=5.0)
	parser.add_argument('--directory', dest='directory', type=str, default=None)
	parser.add_argument('--model', dest='model', type=str, choices=['lstm', 'p3d', 'basic'], default='lstm')
	parser.add_argument('--save_dir', dest='save_dir', default='checkpoints/')

	return parser.parse_args()

def evaluate_ppl(model, val_loader, gamma):
	model.eval()

	cum_loss = 0.
	cum_words = 0.

	with torch.no_grad():
		for data, target in val_loader:
			#loss = -model.forward(data, target).sum()
			losses, probs, log_probs = model.forward(data, target) #removed negative sign!
			#ind_loss = (-1 * torch.pow(1-probs, gamma) * log_probs)
			#batch_loss = ind_loss.sum(dim=0).sum()

			cum_loss += -losses.sum().item()
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
	data_path = 'data/msvd/new_word2id.json' if args.directory == 'data/mpii' else 'data/mpii/new_word2id_mpii.json'
	embed_path = 'data/msvd/word_embed.npy' if args.directory == 'data/mpii' else 'data/mpii/word_embed_mpii.npy'
	if args.model == 'lstm':
		train_dataset = (LSTMMSVDDataset(directory=args.directory, max_frames=args.max_frames, split='train') if args.directory 
					else LSTMMSVDDataset(max_frames=args.max_frames, split='train'))
		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  collate_fn=custom_collate_fn)

		val_dataset = (LSTMMSVDDataset(directory=args.directory, max_frames=args.max_frames, split='val') if args.directory 
					else LSTMMSVDDataset(max_frames=args.max_frames, split='val'))
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

		#model = LSTMCombined('data/msvd/msvd_vocab.json', device=device)	
		model = LSTMCombined(data_path, device=device, pre_embed=embed_path)	
	elif args.model == 'p3d':
		train_dataset = (P3DMSVDDataset(directory=args.directory, max_frames=args.max_frames, split='train') if args.directory 
					else P3DMSVDDataset(max_frames=args.max_frames, split='train'))
		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  collate_fn=custom_collate_fn)

		val_dataset = (P3DMSVDDataset(directory=args.directory, max_frames=args.max_frames, split='val') if args.directory 
					else P3DMSVDDataset(max_frames=args.max_frames, split='val'))
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

		#model = LSTMCombined('data/msvd/msvd_vocab.json', device=device, encoder='p3d')
		model = LSTMCombined(data_path, device=device, pre_embed=embed_path, encoder='p3d')	
	elif args.model == 'basic':
		train_dataset = (LSTMMSVDDataset(directory=args.directory, max_frames=args.max_frames) if args.directory 
					else LSTMMSVDDataset(max_frames=args.max_frames, split='train'))
		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  collate_fn=custom_collate_fn)

		val_dataset = (LSTMMSVDDataset(directory=args.directory, max_frames=args.max_frames) if args.directory 
					else LSTMMSVDDataset(max_frames=args.max_frames, split='val'))
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 4, shuffle=False, collate_fn=custom_collate_fn)

		model = LSTMBasic('data/msvd/msvd_vocab.json', device=device)



	to_filter = ['to_embeddings.weight']
	for name, param in model.named_parameters():
		if param.dim() > 1 and name not in to_filter:
			nn.init.xavier_uniform_(param)
		#if name in to_filter:
			#param.requires_grad = False

	special_params = [param for name, param in model.named_parameters() if name in to_filter]
	base_params = [param for name, param in model.named_parameters() if name not in to_filter]
	param_list = [
		{'params': base_params},
		{'params': special_params, 'lr':args.lr}
	]
	"""param_list = [
		{'params': base_params}
	]"""
	if args.sgd:
		optimizer = optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
	else:
		optimizer = optim.Adam(param_list, lr=args.lr, weight_decay=args.weight_decay)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay_rate, patience=args.patience, verbose=True)
	#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=args.decay_rate)

	model.to(device)
	iteration = 0
	patience = 0
	cum_loss = report_loss = cum_tgt_words = report_tgt_words = cum_examples = report_examples = 0
	valid_ppls = []
	train_ppls = []
	train_losses = []
	x_axis = []
	best_val_ppl = best_train_ppl = float('inf')
	train_time = begin_time = time.time()

	path = join(args.save_dir, str(int(begin_time)))
	make_clean_path(path)
	save_path1 = join(path, '{}_checkpoint1.pth'.format(int(begin_time)))
	save_path2 = join(path, '{}_checkpoint2.pth'.format(int(begin_time)))
	model.save_arguments(path, str(int(begin_time)), vars(args))

	print('Starting training...')
	for epoch in range(args.max_epochs):
		for data, target in train_loader:
			iteration += 1
			model.train()
			optimizer.zero_grad()
			losses, probs, log_probs = model.forward(data, target) #removed negative sign!
			ind_loss = -1 * torch.pow(1-probs, args.gamma) * log_probs
			batch_loss = -losses.sum()
			ind_loss_batch = ind_loss.sum(dim=0).sum()
			loss = ind_loss_batch / args.batch_size
			#batch_loss = losses.sum()
			#loss = batch_loss / args.batch_size

			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), args.clip)
			optimizer.step()

			batch_losses_val = ind_loss_batch.item()
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
				train_loss = cum_loss / cum_examples
				train_ppl = np.exp(cum_loss / cum_tgt_words)
				print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, iteration,
																						 train_loss,
																						 train_ppl,
																						 cum_examples))
				

				print('Starting validation ...')
				valid_ppl = evaluate_ppl(model, val_loader, args.gamma) 
				scheduler.step(train_ppl)

				valid_ppls.append(valid_ppl)
				train_losses.append(train_loss)
				train_ppls.append(train_ppl)
				x_axis.append(iteration)

				plot(x_axis, valid_ppls, join(path, 'valid_ppl.png'), 'Validation Perplexity')
				plot(x_axis, train_losses, join(path, 'train_loss.png'), 'Train Loss')
				plot(x_axis, train_ppls, join(path, 'train_ppl.png'), 'Train Perplexity')
				
				if valid_ppl < best_val_ppl:
					best_val_ppl = valid_ppl
					torch.save(model.state_dict(), save_path1)

				if train_ppl < best_train_ppl:
					best_train_ppl = train_ppl
					torch.save(model.state_dict(), save_path2)

				print('Validation: iter %d, dev. ppl %f, best ppl %f' % (iteration, valid_ppl, best_val_ppl))
				cum_loss = cum_examples = cum_tgt_words = 0.
		#scheduler.step()

def main():
	args = get_arguments()
	train(args)


if __name__ == '__main__':
	main()