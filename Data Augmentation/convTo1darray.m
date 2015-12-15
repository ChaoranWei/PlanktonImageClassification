function a = convTo1darray(mat)
	[row, col, h] = size(mat);
	a = zeros( row * col * h, 1);
	offset = row * col;
	for k = 1:h
		ta = twoDtoOneD( mat(:,:,k) );
		a( 1+(k-1)*offset: k*offset ) = ta(1:offset);
	end
end
