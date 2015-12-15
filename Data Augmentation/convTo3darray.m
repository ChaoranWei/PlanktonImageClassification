function mat = convTo3darray(row, col, h, a )
	offset = row * col;
	for k = 1:h
		tempMat = one2two(row, col, a(1+(k-1)*offset: k*offset ));
		mat(:,:,k) = tempMat;
	endfor
endfunction


