clear;
close all;
clc;

%Improt Data
imdsTrain = imageDatastore("D:\Thesis\LAGrevised","IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imdsTrain,0.7, 0.15, 'randomized' );

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
inputSize = [227, 227];
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'DataAugmentation',imageAugmenter);

%automatically resize the validation images without performing further data augmentation
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);


layersalexnet = [
     imageInputLayer([227 227 3])

     convolution2dLayer(11,96, 'Stride', [4 4], 'Padding', 'same')
     reluLayer
     
     crossChannelNormalizationLayer(5)
     maxPooling2dLayer(3,'Stride',2, 'Padding', 'same')

     groupedConvolution2dLayer(5,128,2, 'Stride', [1 1], 'Padding', [2 2 2 2] )
     reluLayer
     
     crossChannelNormalizationLayer(5)
     maxPooling2dLayer(3,'Stride',2, 'Padding', 'same')
     
     convolution2dLayer(3,384, 'Stride', 1, 'Padding', [1 1 1 1])
     reluLayer
     
     groupedConvolution2dLayer(3,192,2, 'Stride', [1 1], 'Padding', [1 1 1 1] )
     reluLayer
     
     groupedConvolution2dLayer(3,128,2, 'Stride', [1 1], 'Padding', [1 1 1 1] )
     reluLayer
     maxPooling2dLayer(3,'Stride',2, 'Padding', 'same')
     
     fullyConnectedLayer(4096)
     reluLayer
     dropoutLayer(0.5)
     
     fullyConnectedLayer(4096)
     reluLayer
     dropoutLayer(0.5)
     
     fullyConnectedLayer(2)
     softmaxLayer
     classificationLayer];
 
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

%Train the network
net = trainNetwork(augimdsTrain,layersalexnet,options);

%Classify Validation Images
[YPred,probs] = classify(net,augimdsTest);
accuracy = mean(YPred == imdsTest.Labels);

%Display four sample validation images
%with predicted labels and probabilities
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end


 




