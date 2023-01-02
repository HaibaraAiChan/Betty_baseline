import os
import numpy as np
import pandas as pd
from statistics import mean
import argparse
import sys


def data_collect(filename):
	epoch_time =[]
	epoch_wo_REG_Block_gen_time = []
	epoch_wo_to_device_time = []
	epoch_pure_train_time = []
	epoch_load_block_feature_label_time = []
	epoch_block_to_device_time = []
	bucketing_time = []
	metis_partition_time = []
	REG_and_block_gen_time = []
	
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("_buckting time") :
				bucketing_time.append(float(line.split(' ')[-1]))

			if line.startswith("Total (block generation + training)time/epoch") :
				epoch_time.append(float(line.split(' ')[-1]))
			if line.startswith("Training time/epoch"):
				epoch_wo_REG_Block_gen_time.append(float(line.split(' ')[-1]))
			if line.startswith("Training time without block to device /epoch"):
				epoch_wo_to_device_time.append(float(line.split(' ')[-1]))
			if line.startswith("Training time without total dataloading part(pure training) /epoch "):
				epoch_pure_train_time.append(float(line.split(' ')[-1]))
			if line.startswith("load block tensor time/epoch"):
				epoch_load_block_feature_label_time.append(float(line.split(' ')[-1]))
			if line.startswith("block to device time/epoch"):
				epoch_block_to_device_time.append(float(line.split(' ')[-1]))
		
			if line.startswith("block dataloader generation time/epoch"):
				REG_and_block_gen_time.append(float(line.split(' ')[-1]))
			if line.startswith("Metis partitioning:"):
				metis_partition_time.append(float(line.split(' ')[2]))
			
			
			

	print('average bucketing time (each bucket) ',mean(bucketing_time))
	print()
	print('average pure train time per epoch ',mean(epoch_pure_train_time))
	print()	
	print('average features & labels loading time per epoch',mean(epoch_load_block_feature_label_time))
	print('average block to device per epoch',mean(epoch_block_to_device_time))
	print()
	print('average  metis partition  per epoch',mean(metis_partition_time))
	print()
	print('average  REG + block gen per epoch',mean(REG_and_block_gen_time))
	print()

	print('average epoch time w/o REG + block gen',mean(epoch_wo_REG_Block_gen_time))
	print()
	print('average epoch time including REG + block gen',mean(epoch_time))




if __name__=='__main__':
	
	
	file = 'lr_0.001_nb_1_time.log'
	file = 'lr_0.001_nb_2_time.log'
	file = 'lr_0.001_nb_4_time.log'
	file = 'lr_0.001_nb_8_time.log'
	file = 'lr_0.001_nb_16_time.log'
	file = 'lr_0.001_nb_32_time.log'
	data_collect(file)		





