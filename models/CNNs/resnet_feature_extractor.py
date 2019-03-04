import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
import torch.nn as nn

from os.path import join
from PIL import Image
from torchvision import models

from .base_feature_extractor import BaseFeatureExtractor
from utils import transform_img, make_clean_path

size = 224
model = models.resnet152(pretrained=True)
extractor = nn.Sequential(*list(model.children())[:-1])

class ResNetFeatureExtractor(BaseFeatureExtractor):

	def __init__(self):
		super().__init__(extractor, size)

	def preprocess(self, imgs, img_ids, out, name, device):
		outpath = join(out, name)
		make_clean_path(outpath)
		combined_imgs = []
		for img_path in imgs:
			img = Image.open(img_path).convert('RGB')
			img = transform_img(img, size=self.size)
			combined_imgs.append(img)

		x = torch.stack(combined_imgs)
		features = self.extract_features(x, device)

		features = torch.squeeze(torch.squeeze(features, dim=2), dim=2)
		for i in range(features.size()[0]):
			torch.save(features[i], join(outpath, '{}.pt'.format(img_ids[i])))

