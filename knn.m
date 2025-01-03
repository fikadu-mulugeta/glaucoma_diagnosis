clear;
close all;
clc;

%Load the data
table =  readtable('mydatanew.xlsx');
n = size(table,1);

%Split the data using Crossvalidation
hpartition = cvpartition(n,'Holdout',0.3); % Nonstratified partition
idxTrain = training(hpartition);
Train = table(idxTrain,:);
numTrain = size(Train, 1);
idxTest = test(hpartition);
Test = table(idxTest,:);
numTest = size(Test,1);

model = fitcknn(Train,'Remark', 'Distance', 'hamming');

%Deterine misclassification error
rloss = resubLoss(model);

%Construct a cross validate classifier from the model
CVmodel = crossval(model);

%Examine the cross-validation loss, which is the average loss of each cross-validation...
...model when predicting on data that is not used for training.
    kloss = kfoldLoss(CVmodel);

%Apply the model for classification using test data
[lable, score, cost] = predict(model, Test);

%Calculate the Performance
trainError = resubLoss(model);
trainAccuracy = 1-trainError;
testError = loss(model,Test, 'Remark');
testAccuracy = 1-testError;

%Plot a confusion matrix and determine the performance
yt = Test.Remark;
yt = categorical(yt);
predicted = predict(model,Test);
predicted = categorical(predicted);
figure
cm = confusionchart(yt,predicted);
TP = cm.NormalizedValues(1,1);
TN = cm.NormalizedValues(2,2);
FP = cm.NormalizedValues(1,2);
FN = cm.NormalizedValues(2,1);
Accuracy = (TP + TN) / (TP+TN+FP+FN);
Precision = TP / (TP + FP);
Sensitivity = TP / (TP + FN);
Specificity = TN / (TN + FP);
