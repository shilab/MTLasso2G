% Running, simulating, testing and comparing the performance
% of multi-task lassos for once
% Author: Xing Xu @ TTIC
% Last Update: 2011-9-19

function res = runMTLassos(N, K, J, weight_func, k, max_it, is_plot, is_corr, corr_thres1, corr_thres2)
% Input - N, number of samples
%         K, number of labels
%         J, number of features
%         weight_func, function for edge weight
%         k, number of fold for GDCV
%         max_it, max iteration for GDCV
%         is_plot, 1 plot the results, 0 otherwise
% Output - res, cells store results
% Note: default to use correlation graph now
if nargin < 8, is_corr = 0; end

% Simulation setup
group_num = floor(sqrt(K * J / (K + J))) + 1;
sizes1 = randi(floor(sqrt(K)), 1, group_num);
sizes2 = randi(floor(sqrt(J)), 1, group_num);
diff = randi([1 1]);

% Make data
tic
[Y X B] = simuData(sizes1, sizes2, N, K, J, diff);
fprintf('Make data done...\n');
toc

% Normalization
X = X - repmat(mean(X), N, 1);
Y = Y - repmat(mean(Y), N, 1);

% Inits, to reduce computations
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
block_size = floor(N / k);
for i = 1:k
    test_id = (i - 1) * block_size + 1 : min(i * block_size, N);
    training_id = setdiff(1:N, test_id);
    training_X = X(training_id, :);
    training_Y = Y(training_id, :);
    init_val.Bk{k} = pinv(training_X) * training_Y;
end

% % Print the precision and recall value of used graphs
% [mcc1 TP1 TN1 FP1 FN1] = MCC(A1, E1);
% [mcc2 TP2 TN2 FP2 FN2] = MCC(A2, E2);
% fprintf('Precision: %.3f\t%.3f\n', TP1 / (TP1 + FP1), TP2 / (TP2 + FP2));
% fprintf('Recall: %.3f\t%.3f\n', TP1 / (TP1 + FN1), TP2 / (TP2 + FN2));


%% Run standard lasso
% Run Cross-validation gradient descent to choose hyperparameters
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


%% Run standard multi-task lasso
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


%% Run one-graph guided multi-task lasso
% Run Cross-validation gradient descent to choose hyperparameters
lambda_init = lambda_0;
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


%% Run one-graph guided multi-task lasso, here the graph is on features
% Run Cross-validation gradient descent to choose hyperparameters
lambda_init = lambda_0;
gamma1_init = 0;
gamma2_init = 10;
tic
[lambda_1_prime, gamma1_1_prime, gamma2_1_prime, CV_trace_1_prime] = crossValidation(X, Y,...
    G1, G2, k, max_it, lambda_init, gamma1_init, gamma2_init, init_val);
fprintf('4 Cross validation done...\n');
toc

% Main function
tic
Bhat_1_prime = MTLasso_2graph(X, Y, G1, G2, lambda_1_prime, gamma1_1_prime, gamma2_1_prime, init_val);
fprintf('4 Main program done...\n');
toc


%% Run two-graph guided multi-task lasso
% Run Cross-validation gradient descent to choose hyperparameters
lambda_init = (lambda_1 + lambda_1_prime) / 2;
gamma1_init = gamma1_1;
gamma2_init = gamma2_1_prime;
tic
[lambda_2, gamma1_2, gamma2_2, CV_trace_2] = crossValidation(X, Y, G1, G2,...
    k, max_it, lambda_init, gamma1_init, gamma2_init, init_val);
fprintf('5 Cross validation done...\n');
toc

% Main function
tic
Bhat_2 = MTLasso_2graph(X, Y, G1, G2, lambda_2, gamma1_2, gamma2_2, init_val);
fprintf('5 Main program done...\n');
toc

% Plots and calc AUC
area = roc(B, Bhat_0, Bhat_MT, Bhat_1, Bhat_1_prime, Bhat_2, is_plot);

% Return useful values in cells
res = cell(14, 1);
res{1} = B;
res{2} = Bhat_0;
res{3} = Bhat_MT;
res{4} = Bhat_1;
res{5} = Bhat_1_prime;
res{6} = Bhat_2;
res{7} = area;
res{8} = CV_trace_0;
res{9} = CV_trace_MT;
res{10} = CV_trace_1;
res{11} = CV_trace_1_prime;
res{12} = CV_trace_2;
res{13} = G1;
res{14} = G2;

plot_matrix = 1; % control of plot matrix or not
if plot_matrix
    thres = getThreshold(N, K, J);
    B_prime1 = Bhat_0;
    B_prime1(abs(B_prime1) < thres) = 0;
    B_prime2 = Bhat_MT;
    B_prime2(abs(B_prime2) < thres) = 0;
    B_prime3 = Bhat_1;
    B_prime3(abs(B_prime3) < thres) = 0;
    B_prime4 = Bhat_1_prime;
    B_prime4(abs(B_prime4) < thres) = 0;
    B_prime5 = Bhat_2;
    B_prime5(abs(B_prime5) < thres) = 0;
    mcc1 = MCC(B, B_prime1);
    mcc2 = MCC(B, B_prime2);
    mcc3 = MCC(B, B_prime3);
    mcc4 = MCC(B, B_prime4);
    mcc5 = MCC(B, B_prime5);
    
    figure;
    subplot(2, 3, 1);
    colorspy(B);
    title('True B');
    subplot(2, 3, 2);
    colorspy(B_prime1);
    title(sprintf('Lasso Estimated(MCC=%.3f)', mcc1));
    subplot(2, 3, 3);
    colorspy(B_prime2);
    title(sprintf('MTLasso Estimated(MCC=%.3f)', mcc2));
    subplot(2, 3, 4);
    colorspy(B_prime3);
    title(sprintf('MTLasso LG Estimated(MCC=%.3f)', mcc3));
    subplot(2, 3, 5);
    colorspy(B_prime4);
    title(sprintf('MTLasso FG Estimated(MCC=%.3f)', mcc4));
    subplot(2, 3, 6);
    colorspy(B_prime5);
    title(sprintf('MTLasso 2G Estimated(MCC=%.3f)', mcc5));
    maximize(gcf);
end