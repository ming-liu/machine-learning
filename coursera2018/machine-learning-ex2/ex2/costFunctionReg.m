function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta





prediction = sigmoid(X * theta);

p1 = -1 / m * (y' * log(prediction));
p2 = 1 / m * ((1 .- y)' * log(1 .- prediction));
p3 = lambda / (2 * m) * (theta' * theta - theta(1) ^ 2);

J = p1 - p2 + p3;

delta = zeros(size(theta));
thetaSize = size(theta);
delta(2:thetaSize,1) = (lambda / m * theta)(2:thetaSize,1);
grad = 1 / m * (X' * (prediction - y)) + delta;

% =============================================================

end
