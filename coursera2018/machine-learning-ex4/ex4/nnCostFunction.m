function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part1 implemention
A1 = [ones(m,1) X];

Z2 = A1 * Theta1';
A2 = [ones(m,1) sigmoid(Z2)];

Z3 = A2 * Theta2';
h_theta = A3 = sigmoid(Z3);


% size(y) = 5000 * 1,convert to 5000 * 10 .
yk = zeros(m,num_labels);

inx = 1;
for (inx = 1:num_labels );
  yk(find(y == inx),inx) = 1;
end;

J = 1 / m * sum(sum((-yk .* log(h_theta)) - ((1 - yk) .* log(1 - h_theta)))) ;
T1 = Theta1(:,1);
T2 = Theta2(:,1);
regularItem = lambda / (2 * m) * (nn_params' * nn_params - T1'*T1 - T2'*T2) ;
J = J + regularItem;

% Part2 implemention
delta3 = A3 - yk;
delta2 = (delta3 * Theta2) .* sigmoidGradient([ones(m,1) Z2]);  %add 1column to Z2,then next stop,remove delta2(0)
delta2 = delta2(:,2:end);


inx = 1;
D2_ij = zeros(num_labels,hidden_layer_size + 1);
D1_ij = zeros(hidden_layer_size,input_layer_size + 1);
for (inx = 1:m);
  al_2 = A2(inx,:);          % j
  % al_2(1) = 0;
  dl_3 = delta3(inx,:)';     %i
  eachD = dl_3 * al_2;
  D2_ij = D2_ij + eachD;

  al_1 = A1(inx,:);
  dl_2 = delta2(inx,:)';
  eachD1 = dl_2 * al_1;
  D1_ij = D1_ij + eachD1;
end;

regularItem_grad1 = lambda / m * Theta1;
regularItem_grad2 = lambda / m * Theta2;

regularItem_grad1 = [zeros(size(regularItem_grad1,1),1) regularItem_grad1(:,2:end)];
regularItem_grad2 = [zeros(size(regularItem_grad2,1),1) regularItem_grad2(:,2:end)];

Theta2_grad = 1/m * D2_ij + regularItem_grad2;
Theta1_grad = 1/m * D1_ij + regularItem_grad1;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
