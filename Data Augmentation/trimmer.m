function [ im ] = trimmer( image )
%TRIMMER Summary of this function goes here
%   Detailed explanation goes here
im=image{1,1};
for ii=1:9999,
%first check and see if the first row is all the same
m = size(im, 1);
n = size(im,2);
X_row1=im(:,1);
if sum(X_row1~=X_row1(1))==0,
    %Delete the first row
    im=im(:,2:n);
else
    break
end
end

for ii=1:9999,
%first check and see if the first row is all the same
m = size(im, 1);
n = size(im,2);
X_row1=im(:,n);
if sum(X_row1~=X_row1(1))==0,
    %Delete the first row
    im=im(:,1:n-1);
else
    break
end
end

im=im';
for ii=1:9999,
%first check and see if the first row is all the same
m = size(im, 1);
n = size(im,2);
X_row1=im(:,1);
if sum(X_row1~=X_row1(1))==0,
    %Delete the first row
    im=im(:,2:n);
else
    break
end
end

for ii=1:9999,
%first check and see if the first row is all the same
m = size(im, 1);
n = size(im,2);
X_row1=im(:,n);
if sum(X_row1~=X_row1(1))==0,
    %Delete the first row
    im=im(:,1:n-1);
else
    break
end
end

im=im';


end

