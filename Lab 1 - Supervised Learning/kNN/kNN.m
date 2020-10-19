function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% Add your own code here
LPred  = zeros(size(X,1),1);
[z, dim] = size(X);
for i = 1:length(X)
    closest = zeros(2, k);
    longest_dist = 0;
    
    for g = 1:k
        closest(1, g) = LTrain(g, 1);
        closest(2, g) = dist(X(i, 1:dim), XTrain(g, 1:dim));
    end
    longest_dist = max(closest(2, 1:k));
    for b = 1:length(XTrain)
        l = dist(X(i, 1:dim), XTrain(b, 1:dim));
        if l < longest_dist
            [a, arg] = max(closest(2, 1:k));
            closest(1, arg) = LTrain(b, 1);
            closest(2, arg) = l;
            longest_dist = max(closest(2, 1:k));
        end
    end
    LPred(i) = mode(closest(1, 1:k));
end 
end

