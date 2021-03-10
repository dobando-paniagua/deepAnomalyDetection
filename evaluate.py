import numpy as np
import argparse
import matplotlib.pyplot as plt
import string
import re
import math
import os
import torch


thred_b = 0.005
alpha = 1.5
gap_time = 10
valid_start = 8000//gap_time
valid_end = 10000//gap_time
test_start = 10000//gap_time
test_end = 20000//gap_time

valid_anomaly_score = np.zeros((valid_end - valid_start , 1))
test_anomaly_score = np.zeros((test_end - test_start, 1))

matrix_data_path = 'experiments/0001_mscred_20210307_1654/matrix_data/'
test_data_path = matrix_data_path + "test_data/"
reconstructed_data_path = matrix_data_path + "reconstructed_data/"
criterion = torch.nn.MSELoss()

for i in range(valid_start, test_end):
	path_temp_1 = os.path.join(test_data_path, "test_data_" + str(i) + '.npy')
	gt_matrix_temp = np.load(path_temp_1)

	path_temp_2 = os.path.join(reconstructed_data_path, "reconstructed_data_" + str(i) + '.npy')
	reconstructed_matrix_temp = np.load(path_temp_2)
 
	select_gt_matrix = np.array(gt_matrix_temp)[-1][0] #get last step matrix

	select_reconstructed_matrix = np.array(reconstructed_matrix_temp)[0][0]


	#compute number of broken element in residual matrix
	select_matrix_error = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix))
	num_broken = len(select_matrix_error[select_matrix_error >= thred_b])
	


	#print num_broken
	if i < valid_end:
		valid_anomaly_score[i - valid_start] = num_broken
	else:
		test_anomaly_score[i - test_start] = num_broken
valid_anomaly_max = np.max(valid_anomaly_score.ravel())
test_anomaly_score = test_anomaly_score.ravel()


#print(test_anomaly_score)
# plot anomaly score curve and identification result
anomaly_pos = np.zeros(5)
root_cause_gt = np.zeros((5, 3))
anomaly_span = [10, 30, 90]
root_cause_f = open("./data/test_anomaly.csv", "r")
row_index = 0
for line in root_cause_f:
	line=line.strip()
	anomaly_axis = int(re.split(',',line)[0])
	anomaly_pos[row_index] = anomaly_axis/gap_time - test_start - anomaly_span[row_index%3]/gap_time
	root_list = re.split(',',line)[1:]
	for k in range(len(root_list)-1):
		root_cause_gt[row_index][k] = int(root_list[k])
	row_index += 1
root_cause_f.close()

def is_anomaly(value, ranges):
	for _range in ranges:
		if value >= _range[0] and value <= _range[1]:
			return True
	return False


# Getting Precision, Recall, and F1
anomaly_ranges = []
for k in range(len(anomaly_pos)):
	anomaly_ranges.append((anomaly_pos[k], anomaly_pos[k] + anomaly_span[k%3]/gap_time))

true_positives, false_positives, false_negatives = 0, 0, 0
threshold_value = valid_anomaly_max * alpha
for i in range(len(test_anomaly_score)):
	score = test_anomaly_score[i]

	if is_anomaly(i, anomaly_ranges):
		print('Anomaly Detected', test_start + i)
		if score > threshold_value:
			true_positives+= 1
		else:
			false_negatives+= 1
	else:
		if score > threshold_value:
			false_positives+= 1

print('-----------------------------------')
precision = (true_positives / (true_positives+false_positives)) if (true_positives+false_positives) else 0
recall = (true_positives / (true_positives+false_negatives)) if (true_positives+false_negatives) else 0
f1_score = (2*((precision*recall)/(precision+recall))) if (precision+recall) else 0

print('RANGES OF ANOMALY', anomaly_ranges)
print('-- Evaluation Metrics -------------')
print('Threshold', threshold_value)
print('Total tested', len(test_anomaly_score))
print('True Positives', true_positives)
print('False Positives', false_positives)
print('False Negatives', false_negatives)
print('\n')
print('Precision', precision)
print('Recall', recall)
print('F1 Score', f1_score)






fig, axes = plt.subplots()

test_num = test_end - test_start

plt.plot(test_anomaly_score, color = 'black', linewidth = 2)
threshold = np.full((test_num), valid_anomaly_max * alpha)
axes.plot(threshold, color = 'black', linestyle = '--',linewidth = 2)

print('Actual Map')
for k in range(len(anomaly_pos)):
	print(anomaly_pos[k], anomaly_pos[k] + anomaly_span[k%3]/gap_time, anomaly_span[k%3]/gap_time)
	axes.axvspan(anomaly_pos[k], anomaly_pos[k] + anomaly_span[k%3]/gap_time, color='red', linewidth=2)

#labels = [' ', '0e3', '2e3', '4e3', '6e3', '8e3', '10e3']
#axes.set_xticklabels(labels, rotation = 25, fontsize = 20)
plt.xlabel('Test Time', fontsize = 25)
plt.ylabel('Anomaly Score', fontsize = 25)
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.yaxis.set_ticks_position('left')
axes.xaxis.set_ticks_position('bottom')
fig.subplots_adjust(bottom=0.25)
fig.subplots_adjust(left=0.25)
plt.title("MSCRED", size = 25)
plt.savefig('experiments/0001_mscred_20210307_1654/anomaly_score.png')
plt.show()

print(test_anomaly_score)