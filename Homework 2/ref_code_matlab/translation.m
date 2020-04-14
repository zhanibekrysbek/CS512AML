% Translate X by [offset(1) offset(2)] (horizontally and vertically, resp), 
% and return the resulting image Y
% offsets can be negative
function Y = translation(X, offset)
  
  Y = X;
  
  ox = offset(1); 
  oy = offset(2);
  
  [lenx, leny] = size(X);
  
  % General case where ox and oy can be negative 
  % See below for the case where ox and oy are positive (used in this project)
  Y(max(1,1+ox):min(lenx, lenx+ox), max(1,1+oy):min(leny, leny+oy)) ...
     = X(max(1,1-ox):min(lenx, lenx-ox), max(1,1-oy):min(leny, leny-oy));

  % Special case where ox and oy are both positive (used in this project)
  %   Y(1+ox:lenx, 1+oy:leny) = X(1:lenx-ox, 1:leny-oy);

end
