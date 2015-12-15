sample = images{3,2};
subplot(3,3,1);
imshow(sample);

%thresholding using Ostu's Method
level = graythresh(sample);
sample1 = im2bw(sample, 0.95); %above 0.95 will be a good choice
subplot(3,3,2);
imshow(sample1);
%Note: binary conversion looks horrible using popular thresholding
%techniques, in this case, Ostu's Method

%sharpen the image
sample2 = imsharpen(sample);
subplot(3,3,3);
imshow(sample2);

%add noise, but that will make the plankton unclear, so out of question
sample3 = imnoise(sample,'Gaussian');
subplot(3,3,4);
imshow(sample3);

%radon: does not work
sample4 = radon(sample);
subplot(3,3,5);
imshow(sample4);
%imhmax
sample5 = imhmax(sample,40);
subplot(3,3,6);
imshow(sample5);



sample6 = imgaussfilt(sample,1);
subplot(3,3,7);
imshow(sample6);

