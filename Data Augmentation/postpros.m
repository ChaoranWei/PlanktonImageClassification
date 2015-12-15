function mat = postpros(matin)
    mat = int16(matin) + 1;
    mat = genDualImg(mat);
end