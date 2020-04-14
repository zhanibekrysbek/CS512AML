function [stop errors] = crf_test(x, word_list, optimValues)
  
  stop = false;
  w = word_list{1};  
  num_fea = length(w{1}.image);
  num_label = (sqrt(num_fea^2+4*length(x)) - num_fea)/2;
  model.w = reshape(x(1:num_fea*num_label), num_fea, num_label);
  model.T = reshape(x(1+num_fea*num_label:end), num_label, num_label);  

  num_ex = length(word_list);
  errors = [0 0];
  total_letter = 0;
  
  for ex = 1 : num_ex
    word = word_list{ex};
    label_str = decode(word, model);
    letters = cellfun(@(x)(x.label), word, 'UniformOutput', 0);
    score = cat(1, letters{:});
    
    total_letter = total_letter + length(label_str);
    err = sum(score ~= label_str);
    errors(2) = errors(2) + err;
    if err > 0, errors(1) = errors(1) + 1; end    
  end
  
  errors = errors * 100 ./ [num_ex total_letter];
  if nargin == 3
    fprintf('Iter %d: Obj = %g, test errors: word=%g, letter=%g\n', ...
          1+optimValues.iteration, optimValues.fval, errors);
    fid = fopen('monitor.txt', 'a');
    fprintf(fid, '%d %f %f %f\n', 1+optimValues.iteration, optimValues.fval, ...
      errors(1), errors(2));
    fclose(fid);
  end
end
