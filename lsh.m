%load /home/lmb/data/mirflickr25k.mat
load /home/lmb/data/myIAPR.mat
%load /home/lmb/data/FLICKR-25K-split.mat


label_Tr = L_tr;
label_Te = L_te;


% label_Tr = zeros(10, 2173);
% label_Te = zeros(10, 693);

% for i = 1:2173
%     label_Tr(L_tr(i), i) = 1;
% end
% 
% for i = 1:693
%     label_Te(L_te(i), i) = 1;
% end





W_32 = randn(255, 32);
W_64 = randn(255, 64);
W_128 = randn(255, 128);


size(W_32)
size(label_Tr)

Tr_32 = single(W_32' * label_Tr' > 0);
Tr_32(Tr_32 <= 0) = -1;
Tr_32 = [Tr_32'; Tr_32'];

Te_32 = single(W_32' * label_Te' > 0);
Te_32(Te_32 <= 0) = -1;
Te_32 = [Te_32'; Te_32'];

Tr_64 = single(W_64' * label_Tr' > 0);
Tr_64(Tr_64 <= 0) = -1;
Tr_64 = [Tr_64';Tr_64'];

Te_64 = single(W_64' * label_Te' > 0);
Te_64(Te_64 <= 0) = -1;
Te_64 = [Te_64';Te_64'];

Tr_128 = single(W_128' * label_Tr' > 0);
Tr_128(Tr_128 <= 0) = -1;
Tr_128 = [Tr_128';Tr_128'];

Te_128 = single(W_128' * label_Te' > 0);
Te_128(Te_128 <= 0) = -1;
Te_128 = [Te_128';Te_128'];


%save('./mirflickr25k_lsh', 'Tr_32', 'Te_32', 'Tr_64', 'Te_64', 'Tr_128', 'Te_128');
save('./myIAPR_lsh.mat', 'Tr_32', 'Te_32', 'Tr_64', 'Te_64', 'Tr_128', 'Te_128');
%save('./FLICKR-25K-split_lsh', 'Tr_32', 'Te_32', 'Tr_64', 'Te_64', 'Tr_128', 'Te_128');