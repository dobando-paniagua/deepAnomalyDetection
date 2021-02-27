import numpy as np
import argparse
import pandas as pd
import os, sys
import math
import scipy

from scipy.stats import pearsonr
from scipy import spatial
import itertools as it
import string
import re

class SignatureMatrix:

	def __init__(self, step_max, win_size, storage_path='data/'):
		step_max = step_max
		win_size = win_size
		min_time = 0
		max_time = 20000
		gap_time = 10
		
		train_start = 0
		train_end = 8000
		test_start = 8000
		test_end = 20000

		raw_data_path = 'data/synthetic_data_with_anomaly-s-1.csv'
		save_data_path = storage_path
		ts_colname="agg_time_interval"
		agg_freq='5min'

		matrix_data_path = save_data_path + "matrix_data/"
		if not os.path.exists(matrix_data_path):
			os.makedirs(matrix_data_path)


	def generate_signature_matrix_node(self):
		data = np.array(pd.read_csv(self.raw_data_path, header = None), dtype=np.float64)
		sensor_n = data.shape[0]
		# min-max normalization
		max_value = np.max(data, axis=1)
		min_value = np.min(data, axis=1)
		data = (np.transpose(data) - min_value)/(max_value - min_value + 1e-6)
		data = np.transpose(data)

		#multi-scale signature matix generation
		for w in range(len(self.win_size)):
			matrix_all = []
			win = self.win_size[w]
			print ("generating signature with window " + str(win) + "...")
			for t in range(self.min_time, self.max_time, self.gap_time):
				#print t
				matrix_t = np.zeros((sensor_n, sensor_n))
				if t >= 60:
					for i in range(sensor_n):
						for j in range(i, sensor_n):
							#if np.var(data[i, t - win:t]) and np.var(data[j, t - win:t]):
							# rescale by win
							matrix_t[i][j] = np.inner(data[i, t - win:t], data[j, t - win:t])/(win) 
							matrix_t[j][i] = matrix_t[i][j]
				matrix_all.append(matrix_t)
			path_temp = self.matrix_data_path + "matrix_win_" + str(win)
			np.save(path_temp, matrix_all)
			del matrix_all[:]

		print ("matrix generation finish!")

	def generate_train_test_data(self):
		#data sample generation
		print ("generating train/test data samples...")
		matrix_data_path = self.save_data_path + "matrix_data/"

		train_data_path = matrix_data_path + "train_data/"
		if not os.path.exists(train_data_path):
			os.makedirs(train_data_path)
		test_data_path = matrix_data_path + "test_data/"
		if not os.path.exists(test_data_path):
			os.makedirs(test_data_path)

		data_all = []
		# for value_col in value_colnames:
		for w in range(len(self.win_size)):
			#path_temp = matrix_data_path + "matrix_win_" + str(win_size[w]) + str(value_col) + ".npy"
			path_temp = matrix_data_path + "matrix_win_" + str(self.win_size[w]) + ".npy"
			data_all.append(np.load(path_temp))

		train_test_time = [[self.train_start, self.train_end], [self.test_start, self.test_end]]
		for i in range(len(train_test_time)):
			for data_id in range(int(train_test_time[i][0]/self.gap_time), 
									int(train_test_time[i][1]/self.gap_time)):
				#print data_id
				step_multi_matrix = []
				for step_id in range(self.step_max, 0, -1):
					multi_matrix = []
					# for k in range(len(value_colnames)):
					for i in range(len(self.win_size)):
						multi_matrix.append(data_all[i][data_id - step_id])
					step_multi_matrix.append(multi_matrix)

				if data_id >= (self.train_start/self.gap_time + self.win_size[-1]/self.gap_time 
								+ self.step_max) and data_id < (self.train_end/self.gap_time): # remove start points with invalid value
					path_temp = os.path.join(train_data_path, 'train_data_' + str(data_id))
					np.save(path_temp, step_multi_matrix)
				elif data_id >= (self.test_start/self.gap_time) and data_id < (self.test_end/self.gap_time):
					path_temp = os.path.join(test_data_path, 'test_data_' + str(data_id))
					np.save(path_temp, step_multi_matrix)

				del step_multi_matrix[:]

		print ("train/test data generation finish!")


if __name__ == '__main__':
	'''need one more dimension to manage mulitple "features" for each node or link in each time point,
	this multiple features can be simply added as extra channels
	'''
	s_matrix = SignatureMatrix(5, [10,30,60])
	s_matrix.generate_signature_matrix_node()
	s_matrix.generate_train_test_data()