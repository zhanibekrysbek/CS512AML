function accuracy = run_svm_mc(train_data, test_data, c_list)  
  
  [label_tr, feature_tr] = get_LibSVM_data(train_data);
  [label_te, feature_te] = get_LibSVM_data(test_data);
  feature_tr = sparse(feature_tr);
  feature_te = sparse(feature_te);
  
  accuracy = c_list;
  for i = 1:length(c_list)    
    c = c_list(i);
    fprintf(['Training SVM-MC ... c = ' num2str(c) ' ... ']);
    model = liblinear_train(label_tr, feature_tr, ['-q -c ' num2str(c)]);
    fprintf('Done.\n');
  
    [~, res, ~] = liblinear_predict(label_te, feature_te, model); 
    res = res(1);
    fprintf('SVM-MC accuracy: %g\n', res);
    accuracy(i) = res;
  end
end


function [label, feature] = get_LibSVM_data(word_list)
  
  num_ex = length(word_list);
  label = cell(num_ex,1);   
  feature = cell(num_ex,1);
  for i = 1 : num_ex
    w = word_list{i};
    label{i} = cell2mat(cellfun(@(x)(x.label), w, 'UniformOutput', 0));
    tmp = cellfun(@(x)(x.image), w, 'UniformOutput', 0);
    feature{i} = cat(2, tmp{:});
  end
  
  label = cell2mat(label);
  feature = cat(2, feature{:})';  
end
