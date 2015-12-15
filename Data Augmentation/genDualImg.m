%Given a grey-scale file,
function Dmat = genDualImg(mat)
	Dmat =  uint8( abs ( int16(mat) - 255));
end