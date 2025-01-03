clear;
close all;
clc;

%load the saved data store now
imds = imageDatastore('LAGrevised128','IncludeSubfolders',true,'LabelSource','foldernames');
labelsnew = imds.Labels;
numImages = numel(imds.Files);

%partition the datastore
[imdsTrain,imdsValidation, imdsTest] = splitEachLabel(imds,0.7, 0.15, 'randomized');
numImagesTrain = numel(imdsTrain.Files);
numImagesValidation = numel(imdsValidation.Files);
numImagesTest = numel(imdsTest.Files);

%Define Network Architecture
layers = [
    imageInputLayer([128 128 3])
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer  % To speed up training and reduce the sensitivity to network initialization
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3, 32,'Padding','same')
    batchNormalizationLayer  
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3, 32,'Padding','same')
    batchNormalizationLayer  
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3, 32,'Padding','same')
    batchNormalizationLayer  
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3, 32,'Padding','same')
    batchNormalizationLayer  
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3, 32,'Padding','same')
    batchNormalizationLayer  
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3, 32,'Padding','same')
    batchNormalizationLayer  
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
   
    dropoutLayer(0.4)
%     maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(2)
    
    softmaxLayer
    classificationLayer];

%Specify Training Options
options = trainingOptions('adam', ...
     'GradientDecayFactor', 0.9, ...
     'SquaredGradientDecayFactor', 0.999, ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 5, ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 64, ...
    'Shuffle','every-epoch', ...
    'ValidationData', imdsValidation,...
    'ValidationFrequency',1, ...
    'L2Regularization', 0.005, ...
    'Plots', 'training-progress',...
    'Verbose', true,...
    'VerboseFrequency', 1, ...
     plots="training-progress");

%Train the network
net = trainNetwork(imdsTrain, layers, options);

%Classify Validation Images and Compute Accuracy
[YPred, scoresnet] = classify(net,imdsTest);
YTest = imdsTest.Labels;

TestingAccuracy = sum(YPred == YTest)/numel(YTest);

figure 
confusionchart(YTest,YPred)
title("Confusion Matrix")

classNames = net.Layers(end).Classes;
roc = rocmetrics(YTest,scoresnet,classNames);

figure
plot(roc,ShowModelOperatingPoint=false)
legend(classNames)
title("ROC Curve")

aucnet = roc.AUC;
aucnet