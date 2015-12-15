function [ B ] = resizer( A, a,b )
%RESIZER Summary of this function goes here
%   Detailed explanation goes here

B = imresize(A, [a b]);
end

