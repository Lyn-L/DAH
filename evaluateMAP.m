load myWiki.mat;
load mm2018_myWiki_64.mat;

test_I = single(test_I > 0) * 2 -1;
test_T = single(test_T > 0) * 2 -1;
training_I = single(training_I > 0) * 2 -1;
training_T = single(training_T > 0) * 2 -1;

sim_it = training_T * test_I'; sim_ti = training_I * test_T';
map1 = mAP(sim_it,L_tr,L_te,0)
map2 = mAP(sim_ti,L_tr,L_te,0)

prec1 = Precision_topR_wiki(sim_it,L_tr,L_te,0)
prec2 = Precision_topR_wiki(sim_ti,L_tr,L_te,0)