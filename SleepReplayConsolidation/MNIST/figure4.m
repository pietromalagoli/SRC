%% MNIST catastrophic forgetting example, GIDO Class task
%    Load paths
clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../utils'));
addpath(genpath('../sleep'));

%% load mnist_uint8 and create task;
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

[tasks, train_y] = create_class_task(train_x, train_y);     %associate to each pair (x,y) a task number (1-5)
[test_tasks, test_y] = create_class_task(test_x, test_y);

% I reduce the train and test dataset to only the first 2 tasks' elements
indices1 = find(tasks == 1);
indices2 = find(tasks == 2);
indices = sort(union(indices1, indices2));
test_indices1 = find(test_tasks == 1);
test_indices2 = find(test_tasks == 2);
test_indices = sort(union(test_indices1,test_indices2));
train_x = train_x(indices(1:5000),:);
train_y = train_y(indices(1:5000),:);
test_x = test_x(test_indices,:);
test_y = test_y(test_indices,:);
%% Set up neural network

%Initialize net
nn = nnsetup([784 1200 1200 10]);
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

control_nn = nn;
% Train on each task sequentially
% Task 1
control_nn = nntrain(control_nn, train_x(indices1,:), train_y(indices1,:), opts); 
% Task 2
control_nn = nntrain(control_nn, train_x(indices2,:), train_y(indices2,:), opts); 
disp('########## Control NN trained.');

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

NNs = {};
labels = cell(10,1);
nn_pointer = nn;

plot_i = 1;
numiterations = 15000;
layers = [];
task_order = [1,2];

for ii = 1:2
    i=task_order(ii);
    if ii == 1
        indices = indices1;
    else
        indices = indices2;
    end

    % Train on taski
    nn1 = nntrain(nn_pointer, train_x(indices(1:5000),:), train_y(indices(1:5000),:), opts);
    nn_pointer = nn1;

    labels{plot_i} = strcat('Train Task: ', num2str(i));

    % sleep after train on taski
    indices = find(ismember(tasks, task_order(1:ii)));
    sleep_period = numiterations +(ii-1)*numiterations/3;
    sleep_input = create_masked_input(train_x(indices,:), sleep_period, 10);
    [~, norm_constants] = normalize_nn_data(nn_pointer, train_x(indices,:));
    sleep_opts.alpha = norm_constants*alpha_scale; %*2.25;

    % Run NREM
    Snn = sleepnn_old(nn_pointer, sleep_period, t_opts, sleep_opts, ...
                  sleep_input'); % , threshold_scales

    nn_pointer = Snn;
    for k = 1:length(layers)
       Snn.W{layers(k)} = nn1.W{layers(k)};  
    end
    NNs{ii} = {nn1, Snn};
    labels{plot_i + 1} = "Sleep";
    plot_i = plot_i + 2;
end
disp('######## Train+SRC done.')

% Compute and plot the correlations
[Cbefores, Cafters] = compute_activation_correlation_2tasks(nn1,Snn,test_x,test_y);

disp(['Correlations before SRC: ' Cbefores]);
disp(['Correlations after SRC: ' Cafters]);