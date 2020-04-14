function [f, g] = crf_obj(x, word_list, C)
  
  num_ex = length(word_list);
  w = word_list{1};  
  num_fea = length(w{1}.image);
  num_label = (sqrt(num_fea^2+4*length(x)) - num_fea)/2;
  
  W = reshape(x(1:num_fea*num_label), num_fea, num_label);
  T = reshape(x(1+num_fea*num_label:end), num_label, num_label);
  l = zeros(num_label, 100); r = l;
  g_W = zeros(size(W));
  g_T = zeros(size(T));
  f = 0;  
  
  for ex = 1 : num_ex
    word = word_list{ex};
    num_letter = length(word);
    score = cellfun(@(x)((x.image'*W)'), word, 'UniformOutput', 0);
    score = cat(2, score{:});   % num_label * num_letter
        
    % do dynamic programming
    l(:, 1) = 0;    r(:, num_letter) = 0;
    for i = 2 : num_letter
      v = l(:, i-1) + score(:, i-1);
      temp = T + repmat(v, 1, num_label);
      max_tmp = max(temp);
      l(:, i) = log(sum(exp(temp-repmat(max_tmp, num_label, 1)))) + max_tmp;
    end
    for i = num_letter-1 : -1 : 1
      v = r(:, i+1) + score(:, i+1);
      temp = T + repmat(v', num_label, 1);
      max_tmp = max(temp, [], 2);
      r(:, i) = log(sum(exp(temp-repmat(max_tmp, 1, num_label)),2)) + max_tmp;
    end
    
    % Now compute gradient
    l_plus_score = l(:,1:num_letter) + score;
    r_plus_score = r(:,1:num_letter) + score;
    marg = l_plus_score + r_plus_score - score;
    t = max(marg(:,1));
    f = f - log(sum(exp(marg(:,1)-t))) - t;
    marg = exp(marg - repmat(max(marg), num_label, 1));
    marg = marg ./ repmat(sum(marg), num_label, 1);    
    
    for i = 1 : num_letter
      lab = word{i}.label;
      f = f + score(lab, i);
      V = marg(:,i);   V(lab) = V(lab) - 1;
      g_W = g_W - word{i}.image * V';
      if i < num_letter
        next_lab = word{i+1}.label;
        f = f + T(lab, next_lab);
        V = T + repmat(l_plus_score(:,i), 1, num_label) ...
              + repmat(r_plus_score(:,i+1)', num_label, 1);
        V = exp(V - max(max(V)));
        g_T = g_T - V / sum(sum(V));
        g_T(lab, next_lab) = g_T(lab, next_lab) + 1;        
      end
    end
  end
 
  if C > 0
    f = norm(x)^2/2 - f * C / num_ex;
    g = x - [g_W(:); g_T(:)] * (C / num_ex);
  else  % just compute the terms related to log likelihood
    f = f / num_ex;
    g = [g_W(:); g_T(:)] / num_ex;
  end
end
