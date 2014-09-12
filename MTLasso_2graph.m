function B_new = MTLasso_2graph(X, Y, G1, G2,...
                            lambda, gamma1, gamma2, init_val, tol, max_it)

% Two-graph guided multi-task lasso by coordinate-descent algorithm
% Y = X * B + E
% Input:
% X -- feature matrix
% Y -- label/task matrix
% E1 -- prior graph on labels, a binary matrix
% E2 -- prior graph on features, a binary matrix
% weight_option -- weight function
% lambda -- ell_1 regularization weight
% gamma1 -- label fusion regularization weight
% gamma2 -- feature fusion regularization weight
% tol -- convergence criterion
% max_it -- maximum iteration
%
% Output:
% B -- coefficient matrix
%
% Author: Xiaohui Chen (xiaohuic@ece.ubc.ca)
% Last update: Sep-1-2011
K = size(Y, 2);   % K tasks
J = size(X, 2);   % J features

if nargin < 10, max_it = 100; end
if nargin < 9, tol = 1e-4 * J * K; end
if nargin < 8
    B = pinv(X) * Y; % Init
else
    B = init_val.B;
end

% Coordinate Descent
tic
B_new = CDLoop(B, G1.W, G1.C, G1.E, G2.W, G2.C, G2.E, X, Y, lambda, gamma1, gamma2, tol, max_it);
toc