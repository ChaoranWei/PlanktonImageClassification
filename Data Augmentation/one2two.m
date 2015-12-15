function mat = one2two(row, col, a)
#	Row-major form
#	for i = 1:row
#		for j = 1:col
#			mat(i,j) = a(col *(i-1) + j);
#		endfor
#	endfor	

#	Col-major form
	for i = 1:row
		for j = 1:col
			mat(i,j) = a(i + (j-1)*row);
		endfor
	endfor	

endfunction



