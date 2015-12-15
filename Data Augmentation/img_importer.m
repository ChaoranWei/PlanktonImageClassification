% Read in all data

clear all; clc;
file1 = dir('DataSubset');
ND = length(file1);
%images1 = cell(NF,1);

% Note: Make sure to get rid of desktop.ini files... (hence the ND-1, NF-1)
for k = 4 : ND,
    file2=dir(fullfile('DataSubset',file1(k).name));
    NF = length(file2);
    %temp1{k-2}=file1(k).name;
for i= 4 : NF,
    images{i-3,k-3} = imread(fullfile('DataSubset',file1(k).name, file2(i).name));
    %temp2{i-2}=file2(i).name;
end
end
