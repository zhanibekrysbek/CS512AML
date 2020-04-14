from liblinearutil import *
import scipy

train_file = '../../data/train.txt'
test_file = '../../data/test.txt'

output_file = 'baseline_liblinear_results.txt'


def load_data(data_file):
    iStream = open(data_file,'r')
    data = []
    label = []
    word_indicators = []
    for eachline in iStream:
        allparts = eachline.strip().split(' ')
        letter = allparts[1]
        letter_label = ord(letter) - ord('a')
        label.append(letter_label)
        data.append([int(i) for i in allparts[5:]])
        word_indicators.append(int(allparts[3]))
    iStream.close()
    return data, label, word_indicators

def word_wise_acc(p_label, test_label, test_word_indicators):
    word_inds = set(test_word_indicators)
    failed_words = set()
    for i in range(len(p_label)):
        if p_label[i] != test_label[i]:
            failed_words.add(test_word_indicators[i])
    return 1.0 * (len(word_inds) - len(failed_words)) / len(word_inds)



if __name__ == '__main__':
    train_data, train_label, train_word_indicators = load_data(train_file)
    test_data, test_label, test_word_indicators = load_data(test_file)

    train_data_sp = scipy.sparse.csr_matrix(scipy.asarray(train_data))
    test_data_sp = scipy.sparse.csr_matrix(scipy.asarray(test_data))
    train_label = scipy.asarray(train_label)
    test_label = scipy.asarray(test_label)

    ACC_list = {}
    oStream = open(output_file, 'w')
    for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
        c_normalized = c/25953
        prob = problem(train_label, train_data_sp)
        print('c %.10f' % (c))
        param = parameter('-c %.10f -q' % (c_normalized))
        m = train(prob, param)

        p_label, p_acc, p_val = predict(test_label, test_data, m)
        ACC, MSE, SCC = evaluations(test_label, p_label)
        word_acc = word_wise_acc(p_label, test_label, test_word_indicators)
        ACC_list[c] = [ACC, word_acc]
        oStream.write('%.8f %.8f %.8f\n' % (c, word_acc, ACC/100))
        print('word wise acc is %.10f' % (word_acc))
    print(ACC_list)
    oStream.close()
