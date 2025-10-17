numFeatures = 2;    %2 for binary inputs (patches)
loss = 'crossentropy';
layers=[
    featureInputLayer(numFeatures)
    reluLayer()
    fullyConnectedLayer(outputSize=1200)
    reluLayer()
    fullyConnectedLayer(outputSize=1200)
    reluLayer()
];

options= trainingOptions("sgdm",... %https://it.mathworks.com/help/deeplearning/ref/trainingoptions.html

);

