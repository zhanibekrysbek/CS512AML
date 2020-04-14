load train
load test
  
% num_trans_list = [0, 500, 1000, 1500, 200];

num_trans_list = 0;

for i = 1 : length(num_trans_list)
    
  num_transform = num_trans_list(i);
  
  train_data = apply_transform(train_data, num_transform);
  
  test_algo = struct('svm_mc', 0, ...
                     'svm_hmm', 0, ...
                     'crf', 1);
    
  % SVM-MC
  if test_algo.svm_mc
    c_list_svm_mc = 10.^(-2:1);
    c_list_svm_mc = 0.02;
    accuracy_SVM_MC = run_svm_mc(train_data, test_data, c_list_svm_mc);
  end
  
  % SVM-Struct
  if test_algo.svm_hmm
    c_list_svm_hmm = 10.^(-2:2);
%     c_list_svm_hmm = 10.^(2:4);
    accuracy_SVM_hmm = run_svm_hmm(test_data, c_list_svm_hmm);
  end
  
  % CRF
  if test_algo.crf
    c_list_crf = 10.^(0:4);
    c_list_crf = 1000;
    accuracy_CRF = run_crf(train_data, test_data, par, c_list_crf);
  end
  

end
