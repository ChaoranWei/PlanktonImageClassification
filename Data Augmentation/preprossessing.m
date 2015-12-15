a = size(images);
%for i = 1:a(2)
list = dir('DataSubset');
for i = 1:a(2)
    cat = strcat('cat', num2str(i));
    count = 0;
    %for j = 1:a(1)
    for j = 1:a(1)
        b = size(images{j,i});
        if b(1) > 0
            TheImage = images{j,i};
        
            flip = flipdim(TheImage, 2);
            imwrite(flip, strcat('DataSubset/', list(i + 3).name, '/', cat, num2str(count), '.png'));
            count = count + 1;
            
            
            for t = [0.9, 0.8, 0.7]
                trans = affine2d([t 0 0; 0 1 0; 0 0 1]);
                stretch = imwarp(TheImage, trans);
                stretch = blackspace(stretch, [b(1),b(2)]);
               
                imwrite(uint8(stretch), strcat('DataSubset/', list(i + 3).name, '/', cat, num2str(count), '.png'));
                count = count + 1;
                
            end            
            
        
            for k = [20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340]
               
                rot = imrotate(TheImage, k);
                Mrot = ~imrotate(true(size(TheImage)),k);
                rot(Mrot&~imclearborder(Mrot)) = 255;
                rot = imresize(rot,[b(1),b(2)]);
                imwrite(uint8(rot), strcat('DataSubset/', list(i + 3).name, '/', cat, num2str(count), '.png'));
                count = count + 1;
            end
        
            trans = affine2d([1,0,0;0.3,1,0;0,0,1]);
            shear = imwarp(TheImage, trans);
            shear = blackspace(shear, [b(1),b(2)]);
            imwrite(uint8(shear), strcat('DataSubset/', list(i + 3).name, '/', cat, num2str(count), '.png'));
            count = count + 1;
            
        end   
    end
end





