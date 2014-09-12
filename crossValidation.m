% GD cross-validation to choose hyper-parameters
% Author: Xing Xu @ TTIC
% Last Update: 2011-9-19


function [lambda gamma1 gamma2 trace] = crossValidation(X, Y, G1, G2,...
    k, max_it, lambda, gamma1, gamma2, init_val, tol)
% Input - k, number of folds for cross-validation
%         max_it, maximum number of iterations allowed
%         lambda, gamma1, gamma2, initial values
%         tol, allowed difference between two iterations
% Output - trace, trace of parameters and errors
% Other Input and Output have the same meaning with MTLasso_2graph

% Init and settings, important parameters
if nargin < 13, tol = 1e-1; end
stepsize1 = 1e-4;
stepsize2 = 1e-4;
stepsize3 = 1e-4;

% Begin gradient descent for optimizing lambda and gammas
flag_it = 1;
diff = tol + 1; % difference between errors of two iterations
error_old = Inf;
trace = [];
error0 = -1;
while flag_it < max_it && diff > tol
    % finite difference to calc gradient
    deltaX = 1e-4;
    if error0 < 0
        error0 = CV(X, Y, G1, G2, lambda, gamma1, gamma2, k, init_val);
        trace = [trace; [lambda gamma1 gamma2 error0]];
        fprintf('%d %.2f %.2f %.2f %d\n', flag_it, lambda, gamma1, gamma2, error0);
    end
    
    % line search for lambda
    if lambda > 0
        upper_bound = 1000;
        lower_bound = 0;
        error1 = CV(X, Y, G1, G2, lambda + deltaX, gamma1, gamma2, k, init_val);
        gradient1 = (error1 - error0) / deltaX;
        while upper_bound > lower_bound
            lambda_prime = lambda - upper_bound * stepsize1 * gradient1;
            if lambda_prime < lambda / 10
                lambda_prime = lambda / 10;
                upper_bound = 0.9 * lambda / (stepsize1 * gradient1);
            end
            if lambda_prime > 10 * lambda || isnan(error0) % more than 10 times of current value or error inf
                lambda_prime = 10 * lambda;
                upper_bound = 9 * lambda / abs(stepsize1 * gradient1);
            end
            error1_prime = CV(X, Y, G1, G2, lambda_prime, gamma1, gamma2, k, init_val);
            if error1_prime > error0
                upper_bound = floor((upper_bound + lower_bound) / 2);
                continue;
            end
            error0 = error1_prime;
            lambda = lambda_prime;
            break;
        end
        trace = [trace; [lambda gamma1 gamma2 error0]];
        fprintf('%d %.2f %.2f %.2f %d\n', flag_it, lambda, gamma1, gamma2, error0);
    end
    
    % line search for gamma1
    if gamma1 > 0
        upper_bound = 1000;
        lower_bound = 0;
        error2 = CV(X, Y, G1, G2, lambda, gamma1 + deltaX, gamma2, k, init_val);
        gradient2 = (error2 - error0) / deltaX;
        while upper_bound > lower_bound
            gamma1_prime = gamma1 - upper_bound * stepsize2 * gradient2;
            if gamma1_prime < gamma1 / 10
                gamma1_prime = gamma1 / 10;
                upper_bound = 0.9 * gamma1 / (stepsize2 * gradient2);
            end
            if gamma1_prime > 10 * gamma1 || isnan(error0) % more than ten times of current value
                gamma1_prime = 10 * gamma1;
                upper_bound = 9 * gamma1 / abs(stepsize2 * gradient2);
            end
            error2_prime = CV(X, Y, G1, G2, lambda, gamma1_prime, gamma2, k, init_val);
            if error2_prime > error0
                upper_bound = floor((upper_bound + lower_bound) / 2);
                continue;
            end
            error0 = error2_prime;
            gamma1 = gamma1_prime;
            break;
        end
        trace = [trace; [lambda gamma1 gamma2 error0]];
        fprintf('%d %.2f %.2f %.2f %d\n', flag_it, lambda, gamma1, gamma2, error0);
    end
    
    % line search for gamma2
    if gamma2 > 0
        upper_bound = 1000;
        lower_bound = 0;
        error3 = CV(X, Y, G1, G2, lambda, gamma1, gamma2 + deltaX, k, init_val);
        gradient3 = (error3 - error0) / deltaX;
        while upper_bound > lower_bound
            gamma2_prime = gamma2 - upper_bound * stepsize3 * gradient3;
            if gamma2_prime < gamma2 / 10
                gamma2_prime = gamma2 / 10;
                upper_bound = 0.9 * gamma2 / (stepsize3 * gradient3);
            end
            if gamma2_prime > 10 * gamma2 || isnan(error0)
                gamma2_prime = 10 * gamma2;
                upper_bound = 9 * gamma2 / abs(stepsize3 * gradient3);
            end
            error3_prime = CV(X, Y, G1, G2, lambda, gamma1, gamma2_prime, k, init_val);
            if error3_prime > error0
                upper_bound = floor((upper_bound + lower_bound) / 2);
                continue;
            end
            error0 = error3_prime;
            gamma2 = gamma2_prime;
            break;
        end
        trace = [trace; [lambda gamma1 gamma2 error0]];
        fprintf('%d %.2f %.2f %.2f %d\n', flag_it, lambda, gamma1, gamma2, error0);
    end
    
    % update
    diff = error_old - error0; % Allow flucuation here
    error_old = error0;
    flag_it = flag_it + 1;
    if diff < error0 / 100
        break;
    end
end


function error = CV(X, Y, G1, G2, lambda, gamma1, gamma2, k, init_val_all)
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
    init_val_new.B = init_val_all.Bk{k};
    
    % run MTLasso and calc error
    B = MTLasso_2graph(training_X, training_Y, G1, G2, ...
        lambda, gamma1, gamma2, init_val_new);
    y_hat = test_X * B;
    this_error = sum(sum(abs(test_Y - y_hat)));
    error = error + this_error;
end
