function res = runReal(X, Y, weight_func, k, max_it, corr_thres1, corr_thres2)
% Function to run real data
% Output - res, cells store results
% Note: default to use correlation graph now

% Methods to run in this script
run_Lasso = 0;
run_MTLasso = 0;
run_MTLasso_LG = 0;
run_MTLasso_FG = 0;
run_MTLasso_2G = 1;
run_correlation_method = 1;

N = size(X, 1);

% Normalization
X = X - repmat(mean(X), N, 1);
Y = Y - repmat(mean(Y), N, 1);

% Inits and graphs
G1.C = tril(nanFilter(corr(Y)), -1);
inds1 = find(abs(G1.C) > corr_thres1);
G1.E = inds2subs(inds1, size(G1.C)) - 1;
G1.C = G1.C(inds1);

G2.C = tril(nanFilter(corr(X)), -1);
inds2 = find(abs(G2.C) > corr_thres2);
G2.E = inds2subs(inds2, size(G2.C)) - 1;
G2.C = G2.C(inds2);

switch weight_func
    case 'abs'
        G1.W = abs(G1.C);
        G2.W = abs(G2.C);
    case 'minus' % can only be used for correlation graph
        G1.W = abs(G1.C) - corr_thres1;
        G2.W = abs(G2.C) - corr_thres2;
    case 'minus_square'
        G1.W = (abs(G1.C) - corr_thres1) .^ 2;
        G2.W = (abs(G2.C) - corr_thres2) .^ 2;
end
init_val.B = pinv(X) * Y;

% Inits for CV
block_size = floor(N / k);
for i = 1:k
    test_id = (i - 1) * block_size + 1 : min(i * block_size, N);
    training_id = setdiff(1:N, test_id);
    training_X = X(training_id, :);
    training_Y = Y(training_id, :);
    init_val.Bk{k} = pinv(training_X) * training_Y;
end


%% Run standard lasso
% Run Cross-validation gradient descent to choose hyperparameters
if run_Lasso == 1
    lambda_init = 100;
    gamma1_init = 0;
    gamma2_init = 0;
    tic
    [lambda_0, gamma1_0, gamma2_0, CV_trace_0] = crossValidation(X, Y, G1, G2,...
        k, max_it, lambda_init, gamma1_init, gamma2_init, init_val);
    fprintf('1 Cross validation done...\n');
    toc

    % Main function
    tic
    Bhat_0 = MTLasso_2graph(X, Y, G1, G2, lambda_0, gamma1_0, gamma2_0, init_val);
    fprintf('1 Main program done...\n');
    toc
end


%% Run standard multi-task lasso
if run_MTLasso == 1
    lambda_init = 100;
    tic
    [lambda_MT CV_trace_MT] = crossValidation_MTLasso(X, Y, k, max_it, lambda_init, init_val);
    fprintf('2 Cross Validation done...\n');
    toc

    % Main function
    tic
    Bhat_MT = MTLasso(X, Y, lambda_MT, init_val);
    fprintf('2 Main program done...\n');
    toc
end


%% Run one-graph guided multi-task lasso
% Run Cross-validation gradient descent to choose hyperparameters
if run_MTLasso_LG == 1
    lambda_init = 100;
    gamma1_init = 10;
    gamma2_init = 0;
    tic
    [lambda_1, gamma1_1, gamma2_1, CV_trace_1] = crossValidation(X, Y, G1, G2,...
        k, max_it, lambda_init, gamma1_init, gamma2_init, init_val);
    fprintf('3 Cross validation done...\n');
    toc

    % Main function
    tic
    Bhat_1 = MTLasso_2graph(X, Y, G1, G2, lambda_1, gamma1_1, gamma2_1, init_val);
    fprintf('3 Main program done...\n');
    toc
end


%% Run one-graph guided multi-task lasso, here the graph is on features
% Run Cross-validation gradient descent to choose hyperparameters
if run_MTLasso_FG == 1
    lambda_init = 100;
    gamma1_init = 0;
    gamma2_init = 10;
    tic
    [lambda_1_prime, gamma1_1_prime, gamma2_1_prime, CV_trace_1_prime] = crossValidation(X, Y, G1, G2,...
        k, max_it, lambda_init, gamma1_init, gamma2_init, init_val);
    fprintf('4 Cross validation done...\n');
    toc

    % Main function
    tic
    Bhat_1_prime = MTLasso_2graph(X, Y, G1, G2, lambda_1_prime, gamma1_1_prime, gamma2_1_prime, init_val);
    fprintf('4 Main program done...\n');
    toc
end

%% Run two-graph guided multi-task lasso
if run_MTLasso_2G == 1
    lambda_init = 100;
    gamma1_init = 10;
    gamma2_init = 10;
    tic
    [lambda_2, gamma1_2, gamma2_2, CV_trace_2] = crossValidation(X, Y, G1, G2,...
        k, max_it, lambda_init, gamma1_init, gamma2_init, init_val);
    toc

    % Main function
    tic
    Bhat_2 = MTLasso_2graph(X, Y, G1, G2, lambda_2, gamma1_2, gamma2_2, init_val);
    fprintf('Main program done...\n');
    toc
end

%% Run correlation method
if run_correlation_method == 1
    % TBI
end

% Save values
res = cell(12, 1);
% res{1} = Bhat_0;
% res{2} = Bhat_MT;
% res{3} = Bhat_1;
% res{4} = Bhat_1_prime;
res{5} = Bhat_2;
% res{6} = CV_trace_0;
% res{7} = CV_trace_MT;
% res{8} = CV_trace_1;
% res{9} = CV_trace_1_prime;
res{10} = CV_trace_2;
res{11} = G1;
res{12} = G2;
save('real_result', 'res');d