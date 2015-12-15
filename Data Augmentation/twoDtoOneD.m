% %row major form
% function a = twoDtoOneD(mat)
% 	[nrow, ncol] = size(mat);
% 	a = zeros(nrow * ncol,1);
% 	for i = 1:nrow
% 		for j = 1:ncol
% 			a(ncol *(i-1) + j) = mat(i,j);
% 		endfor
% 	endfor
% endfunction

%col major form
function a = twoDtoOneD(mat)
	[nrow, ncol] = size(mat);
	a = zeros(nrow * ncol,1);
	for i = 1:nrow
		for j = 1:ncol
			a(i + (j-1)*nrow) = mat(i,j);
		end
	end
end

	

