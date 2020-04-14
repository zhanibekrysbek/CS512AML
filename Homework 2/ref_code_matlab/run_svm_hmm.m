function accuracy = run_svm_hmm(test_data, c_list)  
  
  accuracy = c_list;
  dir = 'D:\Dropbox\Research\tool_common\code\svm_hmm_windows';
  
  num_ex = length(test_data);
  label = cell(num_ex,1);   
  for i = 1 : num_ex
    w = test_data{i};
    label{i} = cell2mat(cellfun(@(x)(x.label), w, 'UniformOutput', 0));
  end  
  label = cell2mat(label);

  for i = 1:length(c_list)
    c = c_list(i);
    fprintf('============================\n');
    fprintf(['Training SVM-hmm c = ' num2str(c) ' ... ']);
    
    cmd = [dir '\svm_hmm_learn -o 1 -c ' num2str(c) ' train_struct.txt modelfile.txt'];
    [status, result] = system(cmd);
    fprintf('Done.\n');
  
    cmd = [dir '\svm_hmm_classify test_struct.txt modelfile.txt classify.tags']; 
    [status, result] = system(cmd);
    
    prediction = textread('classify.tags');
    accuracy(i) = sum(prediction == label) * 100 / length(label);

    fprintf('SVM-HMM accuracy for c=%g: %g\n', c, accuracy(i));
  end
  
end
