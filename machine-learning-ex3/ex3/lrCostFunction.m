function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

costTerm1 = (-y) .* log(sigmoid(X*theta));
costTerm2 = (1-y) .* log(1 - sigmoid(X*theta));

theta_without_first_param = theta(2:length(theta));
non_regulated_cost = sum(costTerm1 .- costTerm2) / m;
cost_regulation_factor = (lambda / (2*m)) * sum(theta_without_first_param.^2);

J = non_regulated_cost + cost_regulation_factor;

% Originaly copy-paste the regulated cost solution from ex2 as it was already vectorized
% I had a vector param issue that threw a warning about automatic broadcasting
% Didn't want to waste to much time on this issue and I ended up taking this solution from another tutorial
% My solution did work though, I should have probably used repmat to solve it (but wanted to move on to the next section already)
h = sigmoid(X * theta);
grad = (1/m) * (X' * (h - y)) + lambda / m * theta .* [0; ones(size(theta, 1) - 1, 1)];


% =============================================================

grad = grad(:);

end
