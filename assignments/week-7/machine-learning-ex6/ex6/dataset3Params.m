function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
% https://stackoverflow.com/questions/45887813/function-handle-formats-in-octave

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
%{

C1 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma1 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

error = [0 0 10000]; % intialize error to a high value

for CC = C1
	for SS = sigma1
		model = svmTrain(X, y, CC, @(x1, x2) gaussianKernel(x1, x2, SS));
		predictions = svmPredict(model, Xval);
		err = mean(double(predictions ~= yval));
		if err < error(3)
			error(1) = CC;
			error(2) = SS;
			error(3) = err;
		end
	end
end

C = error(1);
sigma = error(2);

%}

C = 1;
sigma =  0.10000;

% =========================================================================
end
