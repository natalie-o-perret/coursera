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

% Part 1: Feedforward

X = [ones(m, 1) X];
yd = eye(num_labels);
yk = yd(y, :);

% Layer 1 to 2:
% a1 = X
z2 = X * Theta1';
a2 = sigmoid(z2);

% Layer 2 to 3:
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
% a3 = h
h = sigmoid(z3);

% No regularization cost
J = ((1 / m) .* sum(sum((-yk) .* log(h) - (1 - yk) .* log(1 - h))));	

% Cost with regularization 
t1jk = Theta1(:, 2 : end);
t2jk = Theta2(:, 2 : end);
reg = (lambda / (2 * m)) * (sum(sumsqr2(t1jk)) + sum(sumsqr2(t2jk))) ;
J = J + reg;

% Part 2: Backpropagation

for t = 1 : m

	% For the input layer, where l=1:
	a1 = [X(t, :)'];

	% For the hidden layers, where l=2:
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)];

	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	ykBool = ([1 : num_labels] == y (t))';

	delta_3 = a3 - ykBool;
	delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)];
	
  % get rid of the bias unit
  delta_2 = delta_2(2 : end);

	% delta_1 is not needed (no error with inputs).

	Theta1_grad = Theta1_grad + delta_2 * a1';
	Theta2_grad = Theta2_grad + delta_3 * a2';
  
end

% Part 3: Regularization 

Theta1_grad = (1 / m) * Theta1_grad + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(: , 2 : end)];
Theta2_grad = (1 / m) * Theta2_grad + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(: , 2 : end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
