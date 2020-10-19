%% This script will help you test out your kNN code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 4; % Change this to load new data 

% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
% plotCase(X,D)

%% Select a subset of the training samples

numBins = 3;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% XBinComb = combineBins(XBins, [1,2,3]);

% Add your own code to setup data for training and test here
XTrain1 = combineBins(XBins, [1,2]);
LTrain1 = combineBins(LBins, [1,2]);
XTest1  = XBins{3};
LTest1  = LBins{3};

XTrain2 = combineBins(XBins, [2,3]);
LTrain2 = combineBins(LBins, [2,3]);
XTest2  = XBins{1};
LTest2  = LBins{1};

XTrain3 = combineBins(XBins, [3,1]);
LTrain3 = combineBins(LBins, [3,1]);
XTest3  = XBins{2};
LTest3  = LBins{2};
best_k = 0;
acc_best = 0;
for i = 1:10
    %% Use kNN to classify data
    %  Note: you have to modify the kNN() function yourself.

    % Set the number of neighbors
    k = i;

    % Classify training data
    LPredTest1 = kNN(XTest1, k, XTrain1, LTrain1);
    LPredTest2 = kNN(XTest2, k, XTrain2, LTrain2);
    LPredTest3 = kNN(XTest3, k, XTrain3, LTrain3);

    LPredTrain1 = kNN(XTrain1, k, XTrain1, LTrain1);
    
    cM1 = calcConfusionMatrix(LPredTest1, LTest1);
    cM2 = calcConfusionMatrix(LPredTest2, LTest2);
    cM3 = calcConfusionMatrix(LPredTest3, LTest3);
    
    acc1 = calcAccuracy(cM1);
    acc2 = calcAccuracy(cM2);
    acc3 = calcAccuracy(cM3);
    
    acc = acc1 + acc2 + acc3; 
    acc = acc/3;
    
    if acc_best < acc 
        best_k = i;
        acc_best = acc;
    end

end
best_k
acc_best

%% Plot classifications
    %  Note: You should not have to modify this code

    if dataSetNr < 4
        plotResultDots(XTrain1, LTrain1, LPredTrain1, XTest1, LTest1, LPredTest1, 'kNN', [], best_k);
    else
        plotResultsOCR(XTest1, LTest1, LPredTest1)
    end