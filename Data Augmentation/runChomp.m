%stacklen = 80;
%[i,j,k] = ind2sub(size(adthmat(:,:,1:stacklen)), find(adthmat(:,:,1:stacklen)));
%[i,j,k] = ind2sub(size(adthmat(:,1:stacklen,:)), find(adthmat(:,1:stacklen,:)));
%[i,j,k] = ind2sub(size(adthmat(500:599,:,1:stacklen)), find(adthmat(500:599,:,1:stacklen)));
%[i,j,k] = ind2sub(size(adthmat(101:200,:,:)), find(adthmat(101:200,:,:)));
%dlmwrite('chompEx.cub', [i,j,k], 'delimiter','\t');
%cmd = ['./chomp chompEx.cub']
%system(cmd);

%randmat = zeros(100);
%randmat = normrnd(0,1,100,100,100) > 0;
%[i,j,k] = ind2sub(size(randmat), find(randmat));
%dlmwrite('chompEx.cub', [i,j,k], 'delimiter','\t');
%cmd = ['./chomp chompEx.cub']
%system(cmd);
function runChomp(adthmat)
[i,j,k] = ind2sub(size(adthmat(:,:,:)), find(adthmat(:,:,:)));
dlmwrite('chompEx.cub', [i,j,k], 'delimiter','\t');
cmd = ['./chomp chompEx.cub']
system(cmd);
end