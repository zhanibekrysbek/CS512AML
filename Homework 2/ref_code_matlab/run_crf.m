function accuracy = run_crf(train_data, test_data, par, c_list)  
  
  accuracy = c_list;
  for i = 1:length(c_list)    
    
    c = c_list(i);
    fprintf(['Training CRF ... c = ' num2str(c) ' ... ']);
    
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

    [model, fval, flag] = fminunc(obj, x0, opt);
    
    [~, errors] = crf_test(model, test_data);
    accuracy(i) = 100 - errors(2);
    fprintf('CRF accuracy for c=%g: %g\n', c, accuracy(i));
  end
end
