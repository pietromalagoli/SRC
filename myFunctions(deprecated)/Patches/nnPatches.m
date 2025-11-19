%% Code to train the NN on the Patches dataset
width = 11;
layers=[
    featureInputLayer(width^2)
    reluLayer()
    fullyConnectedLayer(width^2)
    reluLayer()
    fullyConnectedLayer(4)
    softmaxLayer()
];
net = dlnetwork(layers);  %create the (untrained) network (already initiliazzed)

% I'm using the options they used in Patches/run_patches_softmax.m
options = trainingOptions("sgdm", Momentum=0.5, ... %https://it.mathworks.com/help/deeplearning/ref/trainingoptions.html
    MaxEpochs=15, ...
    MiniBatchSize=2, ...
    LearnRateSchedule = "none", ...     % learning rate is kept constant (default) 
    InitialLearnRate=0.1);
    %Plots="training-progress", ...

% Create the Patches data
num_images = 4;
%width = 11;
overlaps = [12];
num_trials=1;
%acc_after_task1 = zeros(length(overlaps),num_trials);
%acc_after_task2 = zeros(length(overlaps),num_trials);
%acc_after_sleep1 = zeros(length(overlaps),num_trials);


for k = 1:num_trials
for j = 1:length(overlaps)
    [X,y] = create_permutations(width, overlaps(j), num_images);    % generates the Patches

    nn = trainnet(X,y,net,"crossentropy",options);
end
end

%{
[Xtest, ytest] = create_permutations(width, overlaps(j), num_images);    % generates the Patches
yPred = minibatchpredict(nn,Xtest);
ytest
yPred
%}

