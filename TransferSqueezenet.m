clear;
close all;
clc;

%Import Data
imds = imageDatastore('NewData','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds,0.7,0.15, 'randomized');

%Load pretraind network
net = squeezenet;

%display an interactive visualization of the network architecture
% analyzeNetwork(net);

%determine Input size of the network
inputSize = net.Layers(1).InputSize;

%Replace Final Layers
%Extract the layer graph from the trained network.

lgraph = layerGraph(net); 

%Find the names of the two layers to replace.
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

%Determine number of classes
numClasses = numel(categories(imdsTrain.Labels));

%Replace the last conv layer with a new conv layer
newConvLayer =  convolution2dLayer([1, 1],numClasses,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10,"Name",'new_conv');
lgraph = replaceLayer(lgraph,'conv10',newConvLayer);

%Replace the classification Layer with a new one
newClassificatonLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassificatonLayer);

%Augment image datastore before training
%Augmentation Settings
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);

% Resize the images to match the network input layer.
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'DataAugmentation',imageAugmenter);

%automatically resize the validation images without performing further data augmentation
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

%Select training Options
options = trainingOptions('sgdm', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 63, ...
    'InitialLearnRate',3e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',1, ...
    'L2Regularization',0.0005, ...
    'Verbose',true, ...
    'VerboseFrequency', 1, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,lgraph,options);

%Classidy Image
%Classify Validation Images
[YPred,scores] = classify(netTransfer,augimdsTest);

accuracy = mean(YPred == imdsTest.Labels);

