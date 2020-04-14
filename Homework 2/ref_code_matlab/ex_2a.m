load train
rand('state',1);  randn('state', 1);

x = rand(26*128+26^2,1)-0.5;
fid = fopen('model.txt', 'w');
fprintf(fid, '%.20g\n', x);
fclose(fid);

[f, g] = crf_obj(x, train_data, -1);
fprintf('Obj = %g\n', f);
return;
fid = fopen('gradient.txt', 'w');
fprintf(fid, '%g\n', g);
fclose(fid);
