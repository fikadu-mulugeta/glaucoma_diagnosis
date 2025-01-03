clear;
close all;
clc;

%Prepare the dataset
table =  readtable('mydatanew.xlsx');
n = size(table,1);

%Partition the data using crossvalidation
hpartition = cvpartition(n,'Holdout',0.3); % Nonstratified partition
idxTrain = training(hpartition);
Train = table(idxTrain,:);
idxTest = test(hpartition);
Test = table(idxTest,:);

% % Training the model

model = fitcsvm(Train, 'Remark', 'KernelFunction','polynomial');
%Calculate the Performance
trainError = resubLoss(model);
trainAccuracy = 1-trainError;
testError = loss(model,Test, 'Remark');
testAccuracy = 1-testError;

%Plot Confusion Matrix
yt = Test.Remark;
yt = categorical(yt);
predicted = predict(model,Test);
predicted = categorical(predicted);
figure
cm = confusionchart(yt,predicted);

%Calculate the Peformance of the model
TP = cm.NormalizedValues(1,1);
TN = cm.NormalizedValues(2,2);
FP = cm.NormalizedValues(2,1);
FN = cm.NormalizedValues(1,2);
Accuracy = (TP + TN) / (TP+TN+FP+FN);
Precision = TP / (TP + FP);
Sensitivity = TP / (TP + FN);
Specificity = TN / (TN + FP);

%Classify the data using the model
[label,score] = predict(model,Test);