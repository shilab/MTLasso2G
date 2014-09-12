function B = MTLasso(X, Y, lambda, init_val, tol, max_it)

% Standard multi-task lasso by coordinate-descent algorithm
% Y = X * B + E
% Input:
% X -- feature matrix
% Y -- label/task matrix
% lambda -- ell_1/ell_2 regularization weight
% init_val -- optimization initial value
% tol -- convergence criterion
% max_it -- maximum iteration
%
% Output:
% B -- coefficient matrix
% D -- auxillary variables
%
% Author: Xiaohui Chen (xiaohuic@ece.ubc.ca)
%         Xing Xu (xing@ttic.edu)
% Last update: Sep-16-2011
K = size(Y, 2);   % K tasks
J = size(X, 2);   % J features

if nargin < 6, max_it = 1e2; end
if nargin < 5, tol = 1e-2 * J * K; end

% Initialization
if nargin < 4
    B = pinv(X) * Y;
else
    B = init_val.B;
end
D = (sum(abs(B')) / sum(sum(abs(B))))';

flag_it = 0;    % Current number of iterations
diff = tol + 1;
% Coordinate-descent loop
while diff > tol && flag_it < max_it,
    B_old = B;

    % C implementation of updating parameters
    [B, D] = coordinatedescent_MTLasso( B, X, Y, D, lambda );
    
    % Calculate improvement between successive iterations
    diff = sum(sum(abs(B - B_old)));        % Simply vectorized ell_1 norm (easy to compute)
    flag_it = flag_it + 1;
end
