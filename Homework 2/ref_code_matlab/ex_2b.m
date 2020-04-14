load train 
load test

c = 1000;

obj = @(model)crf_obj(model, train_data, c);
test_obj = @(x, optimValues, state)crf_test(x, test_data, optimValues);

x0 = zeros(par.num_fea*par.num_label+par.num_label^2,1);

opt = optimset('display', 'iter-detailed', ...
               'GradObj', 'on', ...
               'MaxIter', 200, ...  % Run maximum 100 iterations
               'TolFun', 1e-4, ...
               'MaxFunEvals', 200, ...  % Allow CRF objective/gradient to be evaluated at most 100 times
               'LargeScale', 'off', ...
               'OutputFcn', test_obj);  % each iteration, calculate and print the test error

[x, fval, flag] = fminunc(obj, x0, opt);
fprintf('Opt obj = %g, avg log-likelihood = %g\n', fval, (-fval + x'*x/2)/c);
return;

fid = fopen('solution.txt', 'w');
fprintf(fid, '%g\n', x);
fclose(fid);

w = test_data{1};  
num_ex = length(test_data);
num_fea = length(w{1}.image);
num_label = (sqrt(num_fea^2+4*length(x)) - num_fea)/2;
model.w = reshape(x(1:num_fea*num_label), num_fea, num_label);
model.T = reshape(x(1+num_fea*num_label:end), num_label, num_label);  

all_let_pred = [];  % predicted letter
all_let_true = [];  % ground truth letter
for ex = 1 : num_ex
  word = test_data{ex};
  label_str = decode(word, model);
  letters = cellfun(@(x)(x.label), word, 'UniformOutput', 0);

  all_let_pred = [all_let_pred; label_str];
  all_let_true = [all_let_true; cat(1, letters{:})];
end
fid = fopen('prediction.txt', 'w');
fprintf(fid, '%d\n', all_let_pred);
fclose(fid);
fid = fopen('ground_truth_label.txt', 'w');
fprintf(fid, '%d\n', all_let_true);
fclose(fid);

accuracy = sum(all_let_pred == all_let_true) * 100 / length(all_let_pred);
fprintf('Letter-wise accuracy = %g\n', accuracy);
