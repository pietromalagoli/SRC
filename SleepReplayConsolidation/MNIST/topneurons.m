%function top_neurons = topneurons(numNeurons,numff)
% numNeurons è 100 nel paper, perché prendono i top 100 neuroni per 
% spiking relativi ad un dato class input
% load mnist_uint8 and create task;
clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../utils'));
addpath(genpath('../MNIST'));
addpath(genpath('../sleep'));

load mnist_uint8;
train_x = double(train_x) / 255;
train_y = double(train_y);

numNeurons = 100;
numff = 25;
numClasses = 10;

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

numiterations = 15000;

%Initialize net
layers = [784 1200 1200 10];
nn = nnsetup(layers); % for MNIST
% Rescale weights for ReLU
for i = 2 : nn.n   
    % Weights - choose between [-0.1 0.1]
    nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)) - 0.5) * 0.01 * 2;
    nn.vW{i - 1} = zeros(size(nn.W{i-1}));
end
nn_temp = nn;
top_neurons = zeros(length(layers)-1,numNeurons,numClasses); % top100 neurons of each layer for each input class
                                                 % top_neurons.shape(#layers,numNeurons,#classes)
[~, labels] = max(train_y, [], 2);
spike_matrices = {zeros(layers(1),numClasses), zeros(layers(2),numClasses),zeros(layers(3),numClasses)}; % Here I stored the spike data
spike_indices = {zeros(layers(1),numClasses), zeros(layers(2),numClasses),zeros(layers(3),numClasses)}; % Here I keep track of the neurons' indices
for j = 1:10 % number of classes in dataset
    disp(['###### Class ' num2str(j)]);
    % Filter train data for the j class
    indices = find(labels == j);
    for i = 1:numff     % numff forward passes to have an high spikes number (in the paper numff=25)
        disp(['Feedforward #' num2str(i)]);
        nn_temp = nnff(nn_temp, train_x(indices(1:5000),:), train_y(1:5000,:)); % for activations in ANN
    end
    disp(['### Feedforwards on class ' num2str(j) ' done.']);
    
    % sleep
    sleep_period = numiterations +fix(j / 2)*numiterations/3;
    sleep_input = create_masked_input(train_x(indices,:), sleep_period, 10);
    [~, norm_constants] = normalize_nn_data(nn_temp, train_x(indices,:));
    sleep_opts.alpha = norm_constants*alpha_scale; %*2.25;
    
    % transmute it to an Snn
    Snn = sleepnn_old(nn_temp, sleep_period, t_opts, sleep_opts, sleep_input');
    for i = 1:length(layers)-1 % number of layers (excluding the classification one i.e. the last one)
        spike_matrices{i}(:, j) = Snn.layers{i}.sum_spikes; % Store spikes for each layer and class
    end
end
%%
% Identify top neurons based on spike counts for each class
for i = 1:length(layers)-1
    normalized_spikes = spike_matrices{i} ./ mean(spike_matrices{i},2,"omitnan"); % normalize with the average over the columns (i.e. over all the input classes for each neuron)
    [~, sortedNeurons] = sort(normalized_spikes,1,'descend','MissingPlacement','last'); % !! must place the NaN values last!!
    top_neurons(i,:,:) = sortedNeurons(1:numNeurons,:);
end
%disp(spike_matrices{end}(1:20,:));
disp(normalized_spikes(1:20,:));
[a, b] = sort(normalized_spikes(1:20,:),1,'descend','MissingPlacement','last');
disp(a);
disp(b);
disp(b(1:10,:));
%disp(top_neurons(end,1:20,:));
matfile = 'topneurons.mat';
varname = 'top_neurons';
save(matfile,varname);
disp(['Variable ' varname ' saved in ' matfile '.']);

