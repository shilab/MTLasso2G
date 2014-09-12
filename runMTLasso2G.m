function res = runMTLasso2G(N, K, J, weight_func, k, max_it, is_plot)
% Input - N, number of samples
%         K, number of labels
%         J, number of features
%         weight_func, function for edge weight
%         k, number of fold for GDCV
%         max_it, max iteration for GDCV
%         is_plot, 1 plot the results, 0 otherwise
% Output - res, cells store results
if nargin < 8
    is_corr = 1; % default to use correlation graph
    corr_thres1 = 0.4;
    corr_thres2 = 0.6;
end

% Simulation setup
group_num = floor(sqrt(K * J / (K + J))) + 1;
sizes1 = randi(floor(sqrt(K)), 1, group_num);
sizes2 = randi(floor(sqrt(J)), 1, group_num);
diff = 1;

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

% Print the precision and recall value of used graphs
% [mcc1 TP1 TN1 FP1 FN1] = MCC(A1, G1.E);
% [mcc2 TP2 TN2 FP2 FN2] = MCC(A2, G2.E);
% fprintf('Precision: %.3f\t%.3f\n', TP1 / (TP1 + FP1), TP2 / (TP2 + FP2));
% fprintf('Recall: %.3f\t%.3f\n', TP1 / (TP1 + FN1), TP2 / (TP2 + FN2));


%% Run two-graph guided multi-task lasso
lambda_init = 100;
gamma1_init = 10;
gamma2_init = 10;
tic
% CV
block_size = floor(N / k);
for i = 1:k
    test_id = (i - 1) * block_size + 1 : min(i * block_size, N);
    training_id = setdiff(1:N, test_id);
    training_X = X(training_id, :);
    training_Y = Y(training_id, :);
    init_val.Bk{k} = pinv(training_X) * training_Y;
end
[lambda_2, gamma1_2, gamma2_2] = crossValidation(X, Y, G1, G2,...
    k, max_it, lambda_init, gamma1_init, gamma2_init, init_val);
% No CV
% fprintf('actually no cv...\n');
% lambda_2 = lambda_init;
% gamma1_2 = gamma1_init;
% gamma2_2 = gamma2_init;
fprintf('Cross validation done...\n');
toc

% Main function
tic
Bhat_2 = MTLasso_2graph(X, Y, G1, G2, lambda_2, gamma1_2, gamma2_2, init_val);
fprintf('Main program done...\n');
toc

% Return useful values in cells
res = cell(4, 1);
res{1} = B;
res{2} = Bhat_2;
res{3} = G1;
res{4} = G2;

if is_plot
    thres = 10 * getThreshold(N, K, J);
    B_prime5 = Bhat_2;
    B_prime5(abs(B_prime5) < thres) = 0;
    mcc5 = MCC(B, B_prime5);
    
    figure;
    subplot(1, 2, 1);
    colorspy(B);
    title('True B');
    subplot(1, 2, 2);
    colorspy(B_prime5);
    title(sprintf('MTLasso 2G Estimated(MCC=%.3f)', mcc5));
    maximize(gcf);
end