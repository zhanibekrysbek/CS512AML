./svm_hmm_learn -c 0.001 ../data/train_struct.txt c_0.001\_model.txt;./svm_hmm_classify ../data/test_struct.txt c_0.001\_model.txt c_0.001\_classify.tags;
./svm_hmm_learn -c 0.01 ../data/train_struct.txt c_0.01\_model.txt;./svm_hmm_classify ../data/test_struct.txt c_0.01\_model.txt c_0.01\_classify.tags;
./svm_hmm_learn -c 0.1 ../data/train_struct.txt c_0.1\_model.txt;./svm_hmm_classify ../data/test_struct.txt c_0.1\_model.txt c_0.1\_classify.tags;
./svm_hmm_learn -c 10000 ../data/train_struct.txt c_10000\_model.txt;./svm_hmm_classify ../data/test_struct.txt c_10000\_model.txt c_10000\_classify.tags;
./svm_hmm_learn -c 1000 ../data/train_struct.txt c_1000\_model.txt;./svm_hmm_classify ../data/test_struct.txt c_1000\_model.txt c_1000\_classify.tags;
./svm_hmm_learn -c 100 ../data/train_struct.txt c_100\_model.txt;./svm_hmm_classify ../data/test_struct.txt c_100\_model.txt c_100\_classify.tags;
./svm_hmm_learn -c 10 ../data/train_struct.txt c_10\_model.txt;./svm_hmm_classify ../data/test_struct.txt c_10\_model.txt c_10\_classify.tags;
./svm_hmm_learn -c 1 ../data/train_struct.txt c_1\_model.txt;./svm_hmm_classify ../data/test_struct.txt c_1\_model.txt c_1\_classify.tags;
