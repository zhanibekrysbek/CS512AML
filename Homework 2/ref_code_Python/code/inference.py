import numpy as np
import time
from itertools import combinations_with_replacement

d = 128
m = 100


def load_params(parameters_file):
	params = np.loadtxt(parameters_file)
	x = np.array(params[:m*d])
	w = np.array(params[m*d:(m*d+26*d)])
	T = np.array(params[(m*d+26*d):])
	x = np.reshape(x, newshape=(m, d))
	w = np.reshape(w, newshape=(26, d))
	T = np.reshape(T, newshape=(26, 26)).T
	return x, w, T


def brute_force(x, w, T):
	m = len(x)
	possible_letters = [i for i in range(26)]
	all_combinations = combinations_with_replacement(possible_letters, m)
	scores_dict = {}
	for each_comb in all_combinations:
		score = 0.0
		for ind, letter in enumerate(each_comb):
			score += np.dot(w[letter,:], x[ind, :])
		for ind in range(len(each_comb)-1):
			score += T[each_comb[ind], each_comb[ind+1]]
		scores_dict[set(each_comb)] = score
	sorted_score = sorted(scores_dict.items(), key=lambda x:x[1], reverse=True)
	return sorted_score[0][0]


# input: x: (m, d), m is # of letters a word has, d is the feature dimension of letter image
# input: w: (26, d), letter weight vector
# input: T: (26, 26), letter-letter transition matrix
# output: letter_indices: (m, 1), letter labels of a word

def dp_infer(x, w, T):
	m = len(x)
	pos_letter_value_table = np.zeros((m, 26), dtype=np.float64)
	pos_best_prevletter_table = np.zeros((m, 26), dtype=np.int)

	# for the position 1 (1st letter), special handling
	# because only w and x dot product is covered and transition is not considered.
	for i in range(26):
		# print(w)
		# print(x)
		pos_letter_value_table[0, i] = np.dot(w[i, :], x[0, :])

	# pos_best_prevletter_table first row is all zero as there is no previous letter for the first letter

	# start from 2nd position
	for pos in range(1, m):
		# go over all possible letters
		for letter_ind in range(26):
			# get the previous letter scores
			prev_letter_scores = np.copy(pos_letter_value_table[pos-1, :])
			# we need to calculate scores of combining the current letter and all previous letters
			# no need to calculate the dot product because dot product only covers current letter and position
			# which means it is independent of all previous letters
			for prev_letter_ind in range(26):
				prev_letter_scores[prev_letter_ind] += T[prev_letter_ind, letter_ind]

			# find out which previous letter achieved the largest score by now
			best_letter_ind = np.argmax(prev_letter_scores)
			# update the score of current positive with current letter
			pos_letter_value_table[pos, letter_ind] = prev_letter_scores[best_letter_ind] + np.dot(w[letter_ind,:], x[pos, :])
			# save the best previous letter for following tracking to generate most possible word
			pos_best_prevletter_table[pos, letter_ind] = best_letter_ind
	letter_indicies = np.zeros((m, 1), dtype=np.int)
	letter_indicies[m-1, 0] = np.argmax(pos_letter_value_table[m-1, :])
	max_obj_val = pos_letter_value_table[m-1, letter_indicies[m-1, 0]]
	# print(max_obj_val)
	for pos in range(m-2, -1, -1):
		letter_indicies[pos, 0] = pos_best_prevletter_table[pos+1, letter_indicies[pos+1, 0]]
	return letter_indicies


if __name__ == '__main__':
	params_file = 'data/decode_input.txt'
	x, w, T = load_params(params_file)
	print(x.shape)
	letter_indicies = dp_infer(x, w, T)
	output_file = 'results/decode_output.txt'
	print(letter_indicies)
	oStream = open(output_file, 'w')
	for i in range(letter_indicies.shape[0]):
		oStream.write('%d\n' % (letter_indicies[i, 0]+1))
	oStream.close()