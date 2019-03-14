import argparse
import os
import torchvision.transforms as transforms

def get_input_output_args():
	parser = argparse.ArgumentParser(description="Video to Image")
	parser.add_argument('-d', dest='directory', type=str, required=True, help='Path to directory with videos')
	parser.add_argument('-o', dest='out', type=str, required=True, help='Path to output directory')
	args =  parser.parse_args()
	return args.directory, args.out

def transform_img(img, size):
	min_size = min(img.size)
	transform_list = []
	transform_list.append(transforms.CenterCrop(min_size))
	transform_list.append(transforms.Resize(size))
	transform_list.append(transforms.ToTensor())
	transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]))
	transform = transforms.Compose(transform_list)

	return transform(img)

def make_clean_path(path):	
	if os.path.exists(path):
		os.system('rm -rf {}'.format(path))
	os.mkdir(path)

def make_safe_path(path):
	if not os.path.exists(path):
		os.mkdir(path)

def custom_collate_fn(batch):
	data = [elem[0] for elem in batch]
	target = [elem[1] for elem in batch]
	return [data, target]