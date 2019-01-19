function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


inner = ((X*theta) - y).^2;
non_regulazied_cost = sum(inner) / (2*m);

theta_without_first_param = theta(2:length(theta));
cost_regulation_factor = (lambda / (2*m)) * sum(theta_without_first_param.^2);

J = sum(inner) / (2*m) + cost_regulation_factor;

num_params = size(X)(2); 

% Create a theta vector with 0 in index 0 to calculate the lambda penalty in a vectorized manner
% Because we expect to not have penalty on j == 0 it makes the calculation easier
penalty_theta = [0; theta(2:num_params, :)];

grad = (1/m) * (X' * ((X*theta) - y));
grad = grad + (lambda/m) * penalty_theta;








% =========================================================================

grad = grad(:);

end
