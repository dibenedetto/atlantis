
import os
import argparse
import numpy as np
from PIL import Image
from skimage.io import imsave

import torch
import torch.nn as nn
from torch.utils import data
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from network import load_model
from utils import *
from dataloader.AtlantisLoader import AtlantisDataSet


MODEL = 'AquaNet'
NAME = MODEL.lower()
SPLIT = 'test'
SPLIT = 'val'
SPLIT = 'test2'
NUM_CLASSES = 56
BATCH_SIZE = 1
NUM_WORKERS = 1
PADDING_SIZE = '768'
DATA_DIRECTORY = './aquanet/../atlantis/'
SAVE_PATH = './aquanet/./result/'+SPLIT+'/'+NAME
RESTORE_FROM = './aquanet/./snapshots/'+NAME+'/epoch29.pth'

def get_arguments():
	parser = argparse.ArgumentParser(description="Network")
	parser.add_argument("--padding-size", type=int, default=PADDING_SIZE,
						help="Comma-separated string with height and width of s")
	parser.add_argument("--model", type=str, default=MODEL,
						help="available options : PSPNet")
	parser.add_argument("--split", type=str, default=SPLIT,
						help="test or val")
	parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
						help="Number of classes to predict (including background).")
	parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
						help="Path to the directory containing the source dataset.")
	parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
						help="Number of images sent to the network in one step.")
	parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
						help="number of workers for multithread dataloading.")
	parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
						help="Where restore model parameters from.")
	parser.add_argument("--save-path", type=str, default=SAVE_PATH,
						help="Path to save result.")
	return parser.parse_args()

args = get_arguments()

def main():

	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)

	model = load_model(args)

	model.eval()
	model.cuda()

	cudnn.enabled = True
	cudnn.benchmark = True

	testloader = data.DataLoader(
			AtlantisDataSet(args.data_dir, split=args.split, padding_size = args.padding_size),
			batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

	for images, labels, name, width, height in testloader:
		src    = images[0].cpu().numpy().transpose(1,2,0)
		images = images.cuda()
		images = F.interpolate(images, [640, 640], mode='bilinear')
		with torch.no_grad():
			_, pred = model(images)

		pred = F.interpolate(pred, src.shape[0:2], mode='bilinear')
		pred = pred.cpu().data[0].numpy().transpose(1, 2, 0)
		pred = np.array(np.argmax(pred, axis=2), dtype=np.uint8)

		#labels = F.interpolate(labels, [src.shape[1], src.shape[0]], mode='bilinear')
		labels = labels.cpu().data[0].numpy()
		labels = np.asarray(labels, dtype=np.uint8)

		fname = f'{args.save_path}/{name[0][:-4]}'

		imsave(f'{fname}_src.png' , src  , check_contrast=False)
		imsave(f'{fname}_pred.png', pred, check_contrast=False)

		pred_col = colorize_mask(pred)
		pred_col.save(f'{fname}_color.png')

		label_col = colorize_mask(labels)
		label_col.save(f'{fname}_gt.png')


if __name__ == '__main__':
	main()
	# os.system('python compute_iou.py --split ' + SPLIT + ' --pred_dir ' + SAVE_PATH )
