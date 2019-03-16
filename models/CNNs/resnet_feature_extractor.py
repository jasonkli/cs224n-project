import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import os
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
		if os.path.exists(outpath):
			return
		make_clean_path(outpath)
		combined_imgs = []
		for i, img_path in enumerate(imgs):
			img = Image.open(img_path).convert('RGB')
			img = transform_img(img, size=self.size)
			combined_imgs.append(img)
			"""plt.imshow((np.transpose(img.cpu().numpy(), (1,2,0)) * 255).astype(np.uint8))
			plt.savefig(join(outpath, '{}.png'.format(img_ids[i])))
			plt.cla()
			plt.clf()"""

		combined_imgs = combined_imgs[:64]
		x = torch.stack(combined_imgs)
		features = self.extract_features(x, device)

		features = torch.squeeze(torch.squeeze(features, dim=2), dim=2)
		for i in range(features.size()[0]):
			np.save(join(outpath, '{}.npy'.format(img_ids[i])), features[i].cpu().numpy())


