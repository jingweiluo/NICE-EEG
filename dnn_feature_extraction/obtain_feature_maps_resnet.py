"""
Obtain ViT features of training and test images in Things-EEG.

using huggingface pretrained ResNet model

"""

import argparse
import torch.nn as nn
import numpy as np
import torch
import os
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification

gpus = [6]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='/data1/labram_data/Things-EEG2/', type=str)
args = parser.parse_args()

print('Extract feature maps ResNet <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model = model.cuda()
model = nn.DataParallel(model, device_ids=[i for i in range(len(gpus))])

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

# Image directories
img_set_dir = os.path.join(args.project_dir, 'Image_set/image_set')
img_partitions = os.listdir(img_set_dir)
for p in img_partitions:
	part_dir = os.path.join(img_set_dir, p)
	image_list = []
	for root, dirs, files in os.walk(part_dir):
		for file in files:
			if file.endswith(".jpg") or file.endswith(".JPEG"):
				image_list.append(os.path.join(root,file))
	image_list.sort()
	# Create the saving directory if not existing
	save_dir = os.path.join(args.project_dir, 'DNN_feature_maps',
		'full_feature_maps', 'resnet', 'pretrained-'+str(args.pretrained), p)
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)

	# Extract and save the feature maps
	# * better to use a dataloader
	for i, image in enumerate(image_list):
		img = Image.open(image).convert('RGB')
		inputs = processor(images=img, return_tensors="pt")
		x = model(**inputs).logits[0]
		feats = x.detach().cpu().numpy()
		file_name = p + '_' + format(i+1, '07')
		np.save(os.path.join(save_dir, file_name), feats)
