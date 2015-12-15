function [ image ] = blackspace( TheImage,dim )

image = ones(dim(1),dim(2));
a = size(TheImage);
for i = 1:dim(1)
    for j = 1:dim(2)
        if i < a(1)   && j < a(2)  && j >= 2 && i >= 2
            
            if TheImage(i,j) == 0
                
                for k = -1:1
                    image(i + k,j) = 255;
                    image(i - k,j) = 255;
                    image(i,j + k) = 255;
                    image(i,j-k) = 255;
                    image(i+k,j+k) = 255;
                    image(i-k,j-k) = 255;
                    image(i-k,j+k) = 255;
                    image(i+k,j-k) = 255;
                end
            elseif j == 1
                image(i + 1,j +1) = 255;
                
            else
                image(i ,j ) = TheImage(i,j);
            end
        else
            image(i,j) = 255;
        end
    end
end

end

