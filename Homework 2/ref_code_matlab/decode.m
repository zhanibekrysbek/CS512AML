function [label_str, max_val] = decode(word, model)
  
  num_letter = length(word);
  num_label = size(model.w, 2);   % weight is #feature * #class
  score = cellfun(@(x)((x.image'*model.w)'), word, 'UniformOutput', 0);
  
  % do dynamic programming
  score = cat(2, score{:});   % num_label * num_letter
    
  % Brute-force  
  % [max_val_bf max_label_bf] = brute_force();

  % Use dynamic programming
  argmax_list = zeros(num_letter, num_label);   
  label_str = zeros(num_letter, 1);
  
  
  [max_val label_str(num_letter)] = max(dp_argmax(num_letter));
  for i = num_letter - 1 : -1 : 1
    label_str(i) = argmax_list(i+1, label_str(i+1));
  end  
  
  function res = dp_argmax(i)
    if i == 1
      res = score(:, 1);
    else
      [res argmax_list(i, :)] = max(repmat(dp_argmax(i-1), 1, num_label) ...                                
                                    + model.T);    
      res = res' + score(:, i);
    end
  end

  function [max_val_bf max_label_bf] = brute_force()  
    
    label = ones(num_letter, 1);
    max_val_bf = -inf;
    while 1
      obj = sum(score(sub2ind(size(score), label', 1:num_letter))) ...
            + sum(model.T(sub2ind(size(model.T), label(1:end-1), label(2:end))));    

      if max_val_bf < obj
        max_val_bf = obj;
        max_label_bf = label;
      end
      
      j = num_letter;
      while j > 0 && label(j) == num_label
        label(j) = 1; 
        j = j - 1;
      end
      if j == 0
        break;
      else
        label(j) = label(j) + 1;
      end    
    end
  end  
end
