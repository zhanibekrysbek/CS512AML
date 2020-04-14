function data = apply_transform(data, num_transform)
  
  if num_transform == 0, return; end
  
  fname = ['trans_data_' num2str(num_transform) '.mat'];
  if exist(fname, 'file')
    load(fname);
    return
  end
  
  fid = fopen('transform.txt', 'r');
  for i = 1 : num_transform
    line = fgetl(fid);
    par = str2num(line(3:end));
    word_id = par(1); 
    w = data{word_id};
    
    switch line(1)
      case 'r'        
        degree = par(2);        
        for j = 1 : length(w)
          w{j}.image = vec(rotation(reshape(w{j}.image, 8, 16), degree));
        end
        
      case 't'        
        offset = par(2:3);
        for j = 1 : length(w)
          w{j}.image = vec(translation(reshape(w{j}.image, 8, 16), offset));
        end

      case 's'
        offset = par(2:3);
        for j = 1 : length(w)
          w{j}.image = vec(shear(reshape(w{j}.image, 8, 16), offset));
        end
    end
    data{word_id} = w;
  end
  fclose(fid);
  save(fname, 'data');
end