'''
This is the evaluation function for optimized parameters (w, T) with given c
It also generate the prediction.txt for q2b
'''

# Imports
import numpy as np

from optimizer import *
from inference import dp_infer


def crf_test(x, test_set):
	"""
	Compute the test accuracy on the list of words (word_list); x is the
	current model (w_y and T, stored as a vector)
	"""

	# x is a vector. so reshape it into w_y and T
	W = np.reshape(x[:128*26], (26, 128))  # each column of W is w_y (128 dim)
	T = np.reshape(x[128*26:], (26, 26))  # T is 26*26
	T = T.transpose()
	labels = []
	y_predicts = []

	# Compute the CRF prediction of test data using W and T
	for word in test_set:
		label, data = word
		labels.append(label)
		y_predicts.append(dp_infer(data, W, T))

	# Compute the test accuracy by comparing the prediction with the ground truth
	l_acc, w_acc = compare(y_predicts, labels)

	# generate the prediction result for Q2b
	# with open('results/prediction.txt', 'w') as f:
	# 	for word in y_predicts:
	# 		for letter in word:
	# 			f.writelines(str(letter[0]+1) + '\n')

	return l_acc, w_acc


def compare(results, label):

	posLetter = 0
	posWord = 0
	results = [np.array([cnm[0] for cnm in result]) for result in results]
	numLetters = sum([len(la) for la in label])
	numWords = len(results)

	for i in range(len(results)):
		if np.all(results[i] == label[i]):
			posWord += 1
		for j in range(len(results[i])):
			if results[i][j] == label[i][j]:
				posLetter += 1

	l_acc = posLetter / numLetters
	w_acc = posWord / numWords

	return l_acc, w_acc

def eval_3(paths, c_list, testSet):

	for iter, c in enumerate(c_list):  # for every c value

		solution = np.loadtxt(paths[iter])  # retrieve the optimized parameters (w, T) learnt by optimizer.py

		print("Evaluate the parameter learned by C = " + str(c))
		model = solution.transpose()  # model is the solution returned by the optimizer

		l_acc, w_acc = crf_test(model, testSet)  # evaluate the performance on test dataset
		print('CRF test letter accuracy for c = {}: {}'.format(c, l_acc))
		print('CRF test word accuracy for c = {}: {}'.format(c, w_acc))


def eval_4(paths, testSet):

	for iter in range(4):  # for every parameter learnt on different tampering set

		solution = np.loadtxt(paths[iter])  # retrieve the optimized parameters (w, T) learnt by optimizer.py

		print("Evaluate the parameter learned by tampering = " + str((iter+1)*500))
		model = solution.transpose()  # model is the solution returned by the optimizer

		l_acc, w_acc = crf_test(model, testSet)  # evaluate the performance on test dataset
		print('CRF test letter accuracy: ' + str(l_acc))
		print('CRF test word accuracy: ' + str(w_acc))


if __name__ == '__main__':

	testPath = 'data/test.txt'
	testSet = readDataset(testPath)
	c_list = [10**x for x in range(3, 4)]
	q3_paths = ['results/parameterc' + str(10**x) + '.txt' for x in range(3, 4)]
	q4_paths = ['results/parameter_t' + str(500*x) + '.txt' for x in range(1, 5)]

	eval_3(q3_paths, c_list, testSet)  # evaluate the parameters learnt on different c Question 3

	# eval_4(q4_paths, testSet)  # evaluate the parameters learnt on different tampering set Question 4







