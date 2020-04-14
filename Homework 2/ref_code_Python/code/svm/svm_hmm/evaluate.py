import numpy as np
import os
import fnmatch

test_file = '../data/test_struct.txt'
result_file = 'results.txt'

def acc(pred, test_labels, test_word_indicators):
    words = set(test_word_indicators)
    fails = set()
    acc = 0.0
    for i in range(len(test_labels)):
        if pred[i] != test_labels[i]:
            fails.add(test_word_indicators[i])
        else:
            acc += 1
    return 1.0 * (len(words)-len(fails)) / len(words), acc/len(test_labels)

if __name__ == '__main__':
    file_list = []
    for filename in os.listdir('.'):
        if fnmatch.fnmatch(filename, '*.tags'):
            file_list.append(filename)

    test_labels = []
    test_word_indicators = []
    iStream = open(test_file, 'r')
    for eachline in iStream:
        allparts = eachline.strip().split(' ')
        test_labels.append(int(allparts[0]))
        test_word_indicators.append(allparts[1])


    oStream = open(result_file, 'w')
    for eachfilename in file_list:
        c_value = float(eachfilename.split('_')[1])
        pred = np.loadtxt(eachfilename)
        word_wise_acc, letter_wise_acc = acc(pred, test_labels, test_word_indicators)

        oStream.write('%.8f %.8f %.8f\n' % (c_value, word_wise_acc, letter_wise_acc))
    oStream.close()
