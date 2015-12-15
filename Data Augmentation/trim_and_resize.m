
% Trim
load('img_importer_output.mat')
a=40;
b=40;
X=[];
y=[];
for j=1:length(images(1,:)),
    for i=1:length(images(:,j)),
        if(sum(size(cell2mat(images(i,j))))>0),
        im1=trimmer(images(i,j));
        im2=resizer(im1,a,b);
        im3=normalizer(im2);
        X=[X; reshape(im3,1,a*b)];
        y=[y; j];
        end
    end
end


% Resize



