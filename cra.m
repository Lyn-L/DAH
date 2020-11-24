load /home/lmb/data/myWiki.mat
%load /home/lmb/data/myIAPR.mat


training = [I_tr, T_tr];
label_tr = L_tr;
test = [I_te, T_te];
label_te = L_te;

[n_tr, h_tr] = size(training);
[n_te, h_te] = size(test);
w = 1;
c = 1;

incomplete_training = reshape([I_tr, zeros(size(T_tr)); zeros(size(I_tr)), T_tr]',[h_tr, w, c, 2*n_tr]);
training_construction = reshape([zeros(size(I_tr)), T_tr; I_tr, zeros(size(T_tr))]',[w,c,h_tr, 2*n_tr]);

incomplete_test = reshape([I_te, zeros(size(T_te)); zeros(size(I_te)), T_te]', [h_te, w, c, 2*n_te]);
test_construction = reshape([zeros(size(I_te)), T_te; I_te, zeros(size(T_te))]', [w,c,h_te,2*n_te]);



%%%%%%%%%%%%%%%%%incomplete data%%%%%%%%%%%%%%%%%
training_I = [I_tr, zeros(size(T_tr))];
training_I = reshape(training_I', [h_tr,w,c,n_tr]);

training_T = [zeros(size(I_tr)), T_tr];
training_T = reshape(training_T', [h_tr,w,c,n_tr]);

test_I = [I_te, zeros(size(T_te))];
test_I = reshape(test_I', [h_te,w,c,n_te]);

test_T = [zeros(size(I_te)), T_te];
test_T = [reshape(test_T', [h_te,w,c,n_te])];


  

input = imageInputLayer([h_tr, w, c], 'name', 'input');
reshape = reshapeLayer('reshape');

layers1 = [
    fullyConnectedLayer(100, 'name', 'fc_11')
    reluLayer('name', 'relu_11')
    fullyConnectedLayer(200, 'name', 'fc_12')
    reluLayer('name','relu_12')
    fullyConnectedLayer(h_tr, 'name', 'fc_13')
    additionLayer(2, 'name', 'add_1')
    ];

%layers1(1).Bias = randn([100 1]) * 0.0001 + 1;
%layers1(3).Bias = randn([200 1]) * 0.0001 + 1;

layers2 = [
    fullyConnectedLayer(100, 'name', 'fc_21')
    reluLayer('name', 'relu_21')
    fullyConnectedLayer(200, 'name', 'fc_22')
    reluLayer('name','relu_22')
    fullyConnectedLayer(h_tr, 'name', 'fc_23')
    additionLayer(2, 'name', 'add_2')
    ];

%layers2(1).Bias = randn([100 1]) * 0.0001 + 1;
%layers2(3).Bias = randn([200 1]) * 0.0001 + 1;



layers3 = [
    fullyConnectedLayer(100, 'name', 'fc_31')
    reluLayer('name', 'relu_31')
    fullyConnectedLayer(200, 'name', 'fc_32')
    reluLayer('name','relu_32')
    fullyConnectedLayer(h_tr, 'name', 'fc_33')
    additionLayer(2, 'name', 'add_3')
    ];

%layers3(1).Bias = randn([100 1]) * 0.0001 + 1;
%layers3(3).Bias = randn([200 1]) * 0.0001 + 1;


lgraph = layerGraph;
lgraph = addLayers(lgraph, input);
lgraph = addLayers(lgraph, reshape);
lgraph = addLayers(lgraph, layers1);
lgraph = addLayers(lgraph, layers2);
lgraph = addLayers(lgraph, layers3);

lgraph = connectLayers(lgraph, 'input', 'reshape');

lgraph = connectLayers(lgraph, 'reshape', 'fc_11');
lgraph = connectLayers(lgraph, 'reshape', 'add_1/in2');

lgraph = connectLayers(lgraph, 'add_1', 'fc_21');
lgraph = connectLayers(lgraph, 'add_1', 'add_2/in2');

lgraph = connectLayers(lgraph, 'add_2', 'fc_31');
lgraph = connectLayers(lgraph, 'add_2', 'add_3/in2')


routputlayer = regressionLayer('name', 'routput');

lgraph = addLayers(lgraph, routputlayer);
lgraph = connectLayers(lgraph, 'add_3', 'routput');


plot(lgraph);


options = trainingOptions('sgdm',...
    'InitialLearnRate', 0.01,...
    'ValidationData',{incomplete_test, test_construction},...
    'MaxEpochs', 150,...
    'MiniBatchSize', 2,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod', 2,....
    'Plots', 'training-progress',...
    'ExecutionEnvironment','cpu',...
    'CheckpointPath', './net',...
    'ValidationPatience', inf);
    

cranet = trainNetwork(incomplete_training,training_construction, lgraph, options); 


training_I = predict(cranet, training_I, 'ExecutionEnvironment', 'cpu');
save('../adapthash/training_I', 'training_I');
save('./training_I', 'training_I');
training_T = predict(cranet, training_T, 'ExecutionEnvironment', 'cpu');
save('../adapthash/training_T', 'training_T');
save('./training_T', 'training_T');
test_I = predict(cranet, test_I, 'ExecutionEnvironment', 'cpu');
save('../adapthash/test_I', 'test_I');
save('./test_I', 'test_I');
test_T = predict(cranet, test_T, 'ExecutionEnvironment', 'cpu');
save('../adapthash/test_T', 'test_T');
save('./test_T', 'test_T');