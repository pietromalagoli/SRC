clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../utils'));
addpath(genpath('../sleep'));

load mnist_uint8;
train_x = double(train_x) / 255;
train_y = double(train_y);

numNeurons = 100;

% Randomly select a subset of training data
rndm_idxs = randperm(size(train_x, 1), 5000);
train_x = train_x(rndm_idxs, :);
train_y = train_y(rndm_idxs, :);

%top_neurons = topneurons(100,25);  
%save('topneurons.mat','top_neurons');

load('topneurons.mat','top_neurons');

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

%% train on the random input data
nn1 = nntrain(nn_pointer, train_x, train_y, opts);
nn_pointer = nn1;
disp('Awake training done.');

%% sleep
sleep_period = 15000; % uguale a num_iterarions in MNIST/run_class_task_summary.m
sleep_input = create_masked_input(train_x, sleep_period, 10);
[~, norm_constants] = normalize_nn_data(nn_pointer, train_x);
sleep_opts.alpha = norm_constants*alpha_scale; %*2.25;

% transmute it to an Snn
Snn = sleepnn_old(nn_pointer, sleep_period, t_opts, sleep_opts, sleep_input');
disp('Sleep done.');
Spointer = Snn;

frs = zeros(length(nn_pointer.a)-2, 4);
fr_stds = zeros(length(nn_pointer.a)-2, 4);

rng_frs = zeros(length(nn_pointer.a)-1, 4);
rng_stds = zeros(length(nn_pointer.a)-1, 4);

for i = 1:length(nn_pointer.a)-1 % number of layers
    spikes = Spointer.layers{i}.sum_spikes;
    for j = 1:4 % number of classes in dataset
        top_spikes = spikes(top_neurons(i,:,j));
        rndm100 = randperm(size(nn_pointer.a{i},2),size(top_neurons,2)); % estrai 100 (size(top_neurons,2)) neuroni
        rndm_spikes = spikes(rndm100,:);
        frs(i,j) = mean(top_spikes,1);
        fr_stds(i,j) = std(top_spikes,1);
        rng_frs(i,j) = mean(rndm_spikes,1);
        rng_stds(i,j) = std(rndm_spikes,1);
    end
end

%% Plot
f = figure();
k = 0;
for i = 2:3 % loop over layers
    for j = 1:4
        k = k + 1;
        subplot(2,4,k)
        
        bar([frs(i,j),rng_frs(i,j)]); hold on;
        er = errorbar([1,2],[frs(i,j),rng_frs(i,j)],[fr_stds(i,j),rng_stds(i,j)]);    
        er.Color = [0 0 0];                            
        er.LineStyle = 'none'; 
        er.LineWidth = 0.8;
        set(gca,'XTickLabel',{num2str(j-1), 'Rand'});
        ylabel('Average firing rate')
        ylim([0,6]);
        if i == 2
             title(strcat('Digit ', num2str(j-1)))
        end
        hold off
    end
end

