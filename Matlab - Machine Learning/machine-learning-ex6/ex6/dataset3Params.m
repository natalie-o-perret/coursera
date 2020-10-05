function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
combinationPairs = getCombinationPairs(values, values);
lastError = 0;
errorCombinationPairs = [];
combinationCount = length(combinationPairs)

for i = 1 : combinationCount
  
  currentCombinationPair = combinationPairs(i, :);
  currentC = currentCombinationPair(1);
  currentSigma = currentCombinationPair(2);

  fprintf("Training with combination %d / %d: {C = %0.2f, sigma = %0.2f}", i, combinationCount, currentC, currentSigma);
  model = svmTrain(X, y, currentC, @(x1, x2) gaussianKernel(x1, x2, currentSigma));
  predictions = svmPredict(model, Xval);
  
  currentError = mean(double(predictions ~= yval)) * 100;
  errorCombinationPairs = [errorCombinationPairs; currentError, currentC, currentSigma];
  fprintf("=> Current error: %0.2f %%\n\n", currentError);
    
end

errors = errorCombinationPairs(:, 1);
[minError, minIndex] = min(errors);
C = errorCombinationPairs(minIndex, 2);
sigma = errorCombinationPairs(minIndex, 3);

fprintf("\nMin Error: %0.2f %% with combination %d: {C = %0.2f, sigma = %0.2f}\n\n", minError, minIndex, C, sigma);

% =========================================================================

end
