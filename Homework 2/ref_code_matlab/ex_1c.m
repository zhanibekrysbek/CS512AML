rand('state',1);  randn('state', 1);

num_label = 26;
num_fea = 16*8;
model.w = rand(num_fea, num_label)-0.5;
model.T = rand(num_label)-0.5;
num_letter = 100;
model.T = model.T';

fid = fopen('decode_input.txt', 'w');
w = cell(num_letter,1);
for i = 1 : num_letter
  w{i}.label = randi(num_label);  
  w{i}.image = rand(num_fea,1)-0.5;
  fprintf(fid, '%.20g\n', w{i}.image);
end
fprintf(fid, '%.20g\n', vec(model.w));
fprintf(fid, '%.20g\n', vec(model.T));
fclose(fid);

[label_str, max_val] = decode(w, model);

fprintf('Max objvalue = %g\n', max_val);
% fid = fopen('decode_output.txt', 'w');
% fprintf(fid, '%d\n', label_str);
% fclose(fid);
