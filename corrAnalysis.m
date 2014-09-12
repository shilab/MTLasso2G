function [R P] = corrAnalysis(X, Y)

[N J] = size(X);
K = size(Y, 2);

% Normalization
X = X - repmat(mean(X), N, 1);
Y = Y - repmat(mean(Y), N, 1);

R = zeros(J, K);
P = zeros(J, K);
for j = 1:J
    for k = 1:K
        [r p] = corrcoef(X(:, j), Y(:, k));
        R(j, k) = r(1, 2);
        P(j, k) = p(1, 2);
    end
end