import argparse
import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset import LSTMMSVDDataset
from models import LSTMCombined
from utils import custom_collate_fn

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_arguments():
	parser = argparse.ArgumentParser(description="Train arguements")
	parser.add_argument('--lr', dest='lr', type=int, default=1e-3, help='Learning rate')
	parser.add_argument('--clip', dest='clip', type=int, default=5, help='Clip gradient')
	parser.add_argument('--max_epochs', dest='max_epochs', type=int, default=30, help="Max epochs")
	parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help="Batch size")
	parser.add_argument('--sgd', dest='sgd', action="store_true", help='Use sgd instead of adam')
	parser.add_argument('--log_iter', dest='log_iter', type=int, default=10)
	parser.add_argument('--val_iter', dest='val_iter', type=int, default=1000)
	parser.add_argument('--max_frames', dest='max_frames', type=int, default=96)
	parser.add_argument('--decay_rate', dest='decay_rate', type=int, default=0.5)
	parser.add_argument('--directory', dest='directory', type=str, default=None)
	parser.add_argument('--model', dest='model', type=str, choices=['lstm'], default='lstm')

	return parser.parse_args()

def train(args):
	if args.model == 'lstm':
		dataset = (LSTMMSVDDataset(directory=args.directory, max_frames=args.max_frames) if args.directory 
					else LSTMMSVDDataset(max_frames=args.max_frames, split='train'))
		loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=custom_collate_fn)
		model = LSTMCombined('data/msvd/msvd_vocab.json', device=device)	

	if args.sgd:
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
	else:
		optimizer = optim.Adam(model.parameters(), lr=args.lr)


	model.to(device)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay_rate)

	for epoch in range(args.max_epochs):
		for data, target in loader:
			model.train()
			optimizer.zero_grad()
			losses = -model.forward(data, target)
			batch_loss = losses.sum()
			loss = batch_loss / args.batch_size
			loss.backward()
			optimizer.step()
			break
		break

def main():
	args = get_arguments()
	train(args)


if __name__ == '__main__':
	main()