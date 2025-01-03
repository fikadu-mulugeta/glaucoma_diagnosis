clear;
close all;
clc;

%Improt Data
imds = imageDatastore('NewData','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds,0.7,0.15, 'randomized');

%Load pretraind network
net = googlenet;

%display an interactive visualization of the network architecture
analyzeNetwork(net);

%determine Input size of the network
inputSize = net.Layers(1).InputSize;

%Replace Final Layers
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

%Find the names of the two layers to replace.
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

%Replace the last fcLayer with a new fc layer

numClasses = numel(categories(imdsTrain.Labels));
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

% Replace the classification layer with a new one without class labels.
%trainNetwork automatically sets the output classes of the layer at training time

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% check that the new layers are connected correctly
%plot the new layer graph and zoom in on the last layers of the network.
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

% Freeze Initial Layers
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

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

%Select training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',6, ...
    'MiniBatchSize', 63, ...
    'Shuffle','every-epoch', ...
    'ValidationData', augimdsValidation,...
    'ValidationFrequency',5, ...
    'L2Regularization', 0.0005, ...
    'Plots', 'training-progress',...
    'Verbose', true,...
    'VerboseFrequency', 1);

%Train the network
net = trainNetwork(augimdsTrain,lgraph,options);

%Classify Validation Images
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels);

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

