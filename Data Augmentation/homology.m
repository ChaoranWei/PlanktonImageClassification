sample = images{3,3};
subplot(2,1,1);
imshow(sample);
%This white noise graph has 300 data points
level = graythresh(sample);
sample1 = im2bw(sample, level);
subplot(2,1,2);
imshow(sample1);
%runChomp(demoMat(i:146,1:1271,1) < 150);

for i = 1:8
    sample = images{i,1};
    level = graythresh(sample);
    sample = im2bw(sample, level);
    runChomp(sample);
end

for i = 1:8
    sample = images{i,2};
    level = graythresh(sample);
    sample = im2bw(sample,level);
    runChomp(sample);
    
end

for i = 1:8
    sample = images{i,3};
    level = graythresh(sample);
    sample = im2bw(sample,level);
    runChomp(sample);
    
end





