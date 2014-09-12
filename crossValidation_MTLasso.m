% GD cross-validation to choose hyper-parameters for standard MTLasso
% Author: Xing Xu @ TTIC
% Last Update: 2011-9-16


function [lambda trace] = crossValidation_MTLasso(X, Y, k, max_it, lambda,...
    init_val, tol)
% Input - k, number of folds for cross-validation
%         max_it, maximum number of iterations allowed
%         lambda, initial values
%         tol, allowed difference between two iterations
% Output - trace, trace of CV

% Init and settings, important parameters
if nargin < 7, tol = 1e-1; end

% Begin gradient descent for optimizing lambda
stepsize = 1e-4;
flag_it = 1;
trace = [];
diff = tol + 1;
error_old = Inf;
error0 = -1;
while flag_it < max_it && diff > tol
    % finite difference to calc gradient
    deltaX = 1e-4;
    if error0 < 0
        error0 = CV(X, Y, lambda, k, init_val);
        trace = [trace; [lambda error0]];
        fprintf('%d %.2f %d\n', flag_it, lambda, error0);
    end
    
    % line search for lambda
    upper_bound = 1000;
    lower_bound = 0;
    error1 = CV(X, Y, lambda + deltaX, k, init_val);
    gradient = (error1 - error0) / deltaX;
    while upper_bound > lower_bound
        lambda_prime = lambda - upper_bound * stepsize * gradient;
        if lambda_prime < lambda / 10
            lambda_prime = lambda / 10;
            upper_bound = 0.9 * lambda / (stepsize * gradient);
        end
        if lambda_prime > 10 * lambda || isnan(error0)
            lambda_prime = 10 * lambda;
            upper_bound = 9 * lambda / abs(stepsize * gradient);
        end
        error1_prime = CV(X, Y, lambda_prime, k, init_val);
        if error1_prime > error0
            upper_bound = floor((upper_bound + lower_bound) / 2);
            continue;
        end
        error0 = error1_prime;
        lambda = lambda_prime;
        break;
    end
    trace = [trace; [lambda error0]];
    fprintf('%d %.2f %d\n', flag_it, lambda, error0);
    
    % update
    diff = error_old - error0;
    error_old = error0;
    flag_it = flag_it + 1;
    if diff < error0 / 100
        break;
    end
end


function error = CV(X, Y, lambda, k, init_val)
% Child function, k-fold cross-validation error for a specific setting
% Input - k, fold number
N = size(X, 1);
block_size = floor(N / k);
error = 0;
for i = 1:k
    test_id = (i - 1) * block_size + 1 : min(i * block_size, N);
    training_id = setdiff(1:N, test_id);
    test_X = X(test_id, :);
    test_Y = Y(test_id, :);
    training_X = X(training_id, :);
    training_Y = Y(training_id, :);
    init_val_k.B = init_val.Bk{k};
    
    % run MTLasso and calc error
    B = MTLasso(training_X, training_Y, lambda, init_val_k);
    y_hat = test_X * B;
    error = error + sum(sum(abs(test_Y - y_hat)));
end
