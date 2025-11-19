%    Load paths
clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../utils'));
addpath(genpath('../sleep'));

load mnist_uint8;
train_x = double(train_x) / 255;
train_y = double(train_y);

%top_neurons = topneurons(100,25);  

%save('topneurons.mat','top_neurons');

load('topneurons.mat','top_neurons');
% I select the top neurons for only tasks 1 and 2 (i.e. labels 0,1 and 2,3
% rispectevely) - top_neurons ha dimensioni (#layers,100,#classes)
topTask1 = top_neurons(:,:,1:2);
topTask2 = top_neurons(:,:,3:4);

%Initialize net
nn = nnsetup([784 1200 1200 10]); % for MNIST
% Rescale weights for ReLU
for i = 2 : nn.n   
    % Weights - choose between [-0.1 0.1]
    nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 0.01 * 2;
    nn.vW{i - 1} = zeros(size(nn.W{i-1}));
end

% ReLU Train
% Set up learning constants     % qua io le modifico per metterle uguali a quelle usate nel paper (table 3)
nn.activation_function = 'relu';
nn.output ='relu';
nn.learningRate = 0.065;
nn.momentum = 0.5;
nn.dropoutFraction = 0.20;
nn.learn_bias = 0;
opts.numepochs =  10;
opts.batchsize = 100;

nn_pointer = nn;

% Filter train data for only task 1 and 2
[~, labels] = max(train_y, [], 2);
train_x = {train_x(ismember(labels, [0, 1]), :) train_x(ismember(labels, [2, 3]), :)};
train_y = {train_y(ismember(labels, [0, 1])) train_y(ismember(labels, [2, 3]))}; 

%{
% Here I have to cut the dimension of the dataset because it has to be a
% multiple of the batch size and by filter wrt the label, it was not
% anymore
numimages = size(train_x,1);
maximages = opts.batchsize * fix(numimages/opts.batchsize);
train_x = train_x(1:maximages,:);
train_y = train_y(1:maximages,:);
%}

% Ho copiato i parametri da MNIST/run_class_task_summary.m senza pensarci
% troppo
x = [239.515363 0.032064 0.003344 14.548273 44.560317 38.046326 55.882454];
inp_rate=x(1);
inc=x(2); 
dec=x(3);
b1=x(4); % 1.1
b2 =x(5);
b3 = x(6);
alpha_scale = x(7);

t_opts = struct;
t_opts.t_ref        = 0.000;
t_opts.threshold    =   1.0;
t_opts.dt           = 0.001;
t_opts.duration     = 0.035;
t_opts.report_every = 0.001;
t_opts.max_rate     = inp_rate;
%t_opts.max_rate=16;

sleep_opts.beta = [b1 b2 b3];
sleep_opts.decay = 0.999; 
sleep_opts.W_inh = 0.0;
sleep_opts.normW = 0;
%sleep_opts.beta = {0.6, 0.3, 0.5};
%sleep_opts.decay = 0.999;
%sleep_opts.alpha = [2.50 2.5 2.5];
sleep_opts.inc = inc;
sleep_opts.dec = dec;
sleep_opts.DC = 0.0;

%% train sequentially on Task 1 and 2, in this order
for i =1:2 
    disp(['## Awake training of Task ' num2str(i) ' done.']);
    nn1 = nntrain(nn_pointer, train_x{i}(1:5000,:), train_y{i}(1:5000,:), opts);
    nn_pointer = nn1;
end
disp('Awake training done.');

%% sleep
sleep_period = 15000; % uguale a num_iterarions in MNIST/run_class_task_summary.m
sleep_input = create_masked_input(train_x{i}, sleep_period, 10);
[~, norm_constants] = normalize_nn_data(nn_pointer, train_x{i});
sleep_opts.alpha = norm_constants*alpha_scale; %*2.25;

% transmute it to an Snn
Snn = sleepnn_old(nn_pointer, sleep_period, t_opts, sleep_opts, sleep_input');
disp('Sleep done.');
Spointer = Snn;

%{
preInput = {}; % cell for storing the results (awake)
postInput = {}; % (sleep)
%}

%% To do: compute input to task-specific neurons
for i = 2:3 % we consider the 2 hidden layers
    % Awake
    activations = nn_pointer.a{i-1} * nn_pointer.W{i-1}';
    % Task 1 
    activationsTask1 = activations(:,topTask1(i,:,:)); % topTask1(i,:,:) sono gli indici dei top 100 neuroni del layer i relativi a task 1 (class number 1,2)
    preInputTask1 = mean(activationsTask1,1);  % Prendo la media dell'activation per ogni neurone rispetto al minibatch size (dimensione 1 di activations)
    % Task 2
    activationsTask2 = activations(:,topTask2(i,:,:)); 
    preInputTask2 = mean(activationsTask2,1);  
    
    % Sleep
    Sactivations = Spointer.a{i-1} * Spointer.W{i-1}';
    % Task 1 
    SactivationsTask1 = Sactivations(:,topTask1(i,:,:)); 
    postInputTask1 = mean(SactivationsTask1,1);  
    % Task 2
    SactivationsTask2 = Sactivations(:,topTask2(i,:,:)); 
    postInputTask2 = mean(SactivationsTask2,1);  

    if i == 2
        HL1 = {preInputTask1, postInputTask1, preInputTask2, postInputTask2};
    else
        HL2 = {preInputTask1, postInputTask1, preInputTask2, postInputTask2};
    end

end
%save('Figure5.mat','preInput','postInput');
matfile = 'Figure5.mat';
varname1 = 'HL1';
varname2 = 'HL2';
save(matfile,varname1,varname2);
disp(['Data saved in variables ' varname1 varname2 ' in file ' matfile '.']);