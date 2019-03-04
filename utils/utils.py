import argparse
import torchvision.transforms as transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

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
	transform_list.append(transforms.Normalize(mean=MEAN, std=STD))
	transform = transforms.Compose(transform_list)

	return transform(img)