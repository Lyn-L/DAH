%load /home/lmb/data/mirflickr25k.mat
%load /home/lmb/data/myIAPR.mat
load /home/lmb/data/FLICKR-25K-split.mat

% val_size = 200;
% anchor_size = 256;
% [n_t,~]  = size(T_tr);
% [n_i,~] = size(I_tr);
% 
% ind_t = randperm(n_t);
% val_label_index_t     = ind_t(1:val_size);
% anchor_samples_t      = ind_t(val_size+1:val_size+anchor_size);
% val_label_index_t     = val_label_index_t';
% anchor_samples_t      = anchor_samples_t';
% val_t            = T_tr(val_label_index_t,:);
% anchor_t      = T_tr(anchor_samples_t,:);
% k_val_t  = sqdist(val_t',anchor_t');
% sigma_t       = mean(mean(k_val_t,2));
% k_val_t  = exp(-k_val_t/(2*sigma_t));
% mvec_t        = mean(k_val_t);
% %k_val_gist  = k_val_gist-repmat(mvec,n,1);
% clear k_val_t val_t
% 
% ind_i = randperm(n_i);
% val_label_index_i     = ind_i(1:val_size);
% anchor_samples_i      = ind_i(val_size+1:val_size+anchor_size);
% val_label_index_i     = val_label_index_i';
% anchor_samples_i      = anchor_samples_i';
% val_i            = I_tr(val_label_index_i,:);
% anchor_i      = I_tr(anchor_samples_i,:);
% k_val_i  = sqdist(val_i',anchor_i');
% sigma_i       = mean(mean(k_val_i,2));
% k_val_i  = exp(-k_val_i/(2*sigma_i));
% mvec_i        = mean(k_val_i);
% %k_val_gist  = k_val_gist-repmat(mvec,n,1);
% clear k_val_i val_i
% 
% 
% [n_tr,~]       = size(T_tr);
% T_tr      = sqdist(T_tr',anchor_t');
% T_tr      = exp(-T_tr/(2*sigma_t));
% T_tr      = T_tr-repmat(mvec_t,n_tr,1);
% [n_te,~]       = size(T_te);
% T_te       = sqdist(T_te',anchor_t');
% T_te       = exp(-T_te/(2*sigma_t));
% T_te       = T_te-repmat(mvec_t,n_te,1);
% 
% 
% [n_t,~]       = size(I_tr);
% I_tr      = sqdist(I_tr',anchor_i');
% I_tr      = exp(-I_tr/(2*sigma_i));
% I_tr      = I_tr-repmat(mvec_i,n_t,1);
% [n_te,~]       = size(I_te);
% I_te       = sqdist(I_te',anchor_i');
% I_te       = exp(-I_te/(2*sigma_i));
% I_te       = I_te-repmat(mvec_i,n_te,1);




%W_T = randn(10, 128);
%T_te = T_te * W_T;
%T_tr = T_tr * W_T;

[h, w] = size(L_tr);
for i = 1:h
    temp = L_tr(i, :);
    len = sum(temp);
    L_tr(i, 1:len) = find(L_tr(i,:) == 1) - 1;
    L_tr(i, len+1:end) = 0;
    L_tr(i, len+1) = -1;
end

[h, w] = size(L_te);
for i = 1:h
    temp = L_te(i, :);
    len = sum(temp);
    L_te(i, 1:len) = find(L_te(i,:) == 1) - 1; 
    L_te(i, len+1:end) = 0;
    L_te(i, len+1) = -1;
end




training_incomplete = [I_tr, zeros(size(T_tr)); zeros(size(I_tr)), T_tr];
training_complete = [zeros(size(I_tr)), T_tr; I_tr, zeros(size(T_tr))];
training_full = [I_tr, T_tr; I_tr, T_tr];
training_label = [L_tr;L_tr];

test_incomplete = [I_te, zeros(size(T_te)); zeros(size(I_te)), T_te];
test_complete = [zeros(size(I_te)), T_te; I_te, zeros(size(T_te))];
test_full = [I_te, T_te; I_te, T_te];
test_label = [L_te;L_te];

training_I = [I_tr, zeros(size(T_tr))];
training_T = [zeros(size(I_tr)), T_tr];
test_I = [I_te, zeros(size(T_te))];
test_T = [zeros(size(I_te)), T_te];


%save('./myIAPR.mat', 'training_incomplete', 'training_complete', 'training_full', 'training_label', 'test_incomplete', 'test_complete','test_full','test_label','training_I','training_T','test_I','test_T','I_te','I_tr','T_te','T_tr','L_tr','L_te');
%save('./mirflickr25k.mat', 'training_incomplete', 'training_complete', 'training_full', 'training_label', 'test_incomplete', 'test_complete','test_full','test_label','training_I','training_T','test_I','test_T','I_te','I_tr','T_te','T_tr','L_tr','L_te');
save('./FLICKR-25K-split.mat', 'training_incomplete', 'training_complete', 'training_full', 'training_label', 'test_incomplete', 'test_complete','test_full','test_label','training_I','training_T','test_I','test_T','I_te','I_tr','T_te','T_tr','L_tr','L_te');
