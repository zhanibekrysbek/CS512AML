% Rotate X by alpha degrees (angle) in a counterclockwise direction around its center point.
% This may enlarge the image.
% So trim the result back to the original size, around its center point.
function Y = rotation(X, alpha)

  Y = imrotate(X, alpha); % Python counterpart: scipy.misc.imrotate
  
  [lenx1, lenx2] = size(X);
  [leny1, leny2] = size(Y);
  
  % Trim the result back to the original size, around its center point.
  fromx = floor((leny1 + 1 - lenx1)/2);
  fromy = floor((leny2 + 1 - lenx2)/2);
  
  Y = Y(fromx:fromx+lenx1-1, fromy:fromy+lenx2-1);
  
  % imrotate could pad 0 at some pixels.
  % At those pixels, fill in the original values
  idx = find(Y == 0);
  Y(idx) = X(idx);

end
