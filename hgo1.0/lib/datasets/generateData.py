# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn

from datasets.HgoDataset import HgoDataset

def generate_dataset(dataset_name, cfg, period):

	if dataset_name == 'hgo2019':
		return HgoDataset('hgo2019',cfg,period)

	else:
		raise ValueError('generateData.py: dataset %s is not support yet'%dataset_name)
