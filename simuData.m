% Data generation for simulation of Two-graph guided Multi-task Lasso
% To use, run function simuData
% Note: The edge weight on graph is the absolute value of correlation
% Author: Xing Xu @ TTIC
% Last Update: 2011-9-19


function [Y X B] = simuData(sizes1, sizes2, N, K, J, diff)
% Main function
% Y = X * B + noise, here noise is Gaussian
% Input - s1, percentage of used edges in label graph
%         sizes1, a vector indicate the size of each group
%         s2, sizes2, the same with above but in feature graph
%         N, number of samples
%         K, number of gene traits
%         J, number of SNPs
%         diff, see comments in sampleMatrix, recommend 5~10
% Note: now B are binary matrix

% Assertions about parameters
assert(length(sizes1) == length(sizes2),...
    'sizes1 and sizes2 should have the same size');

% Sample label groups
groups1 = sampleGroups(K, sizes1);

% Sample feature groups
groups2 = sampleGroups(J, sizes2);

% Sample X
C = sampleCovariateMatrix(J, groups2, diff);
X = rand(N, J) * C;

% Get B from groups1 and groups2, which sizes should be identical
B = zeros(J, K);
for i = 1:length(groups1)
    B(groups2{i}, groups1{i}) = 1;
end

% Get Y
noise = randn(N, K) * 1e-3;
Y = X * B + noise;


function C = sampleCovariateMatrix(l, groups, diff)
% Child function, sample a matrix based on groups, columns in the same
% groups have larger correlation, entries are in range (0, 1)
% input - l, size of matrix wanted
%         groups, groups sampled from sampleGroups
%         diff, the difference between two nodes in the same group and not
%               in the same group, if =1 then no difference, recommend 5~10
C = 0.001 * randn(l, l); % covariate matrix
for i = 1:length(groups)
    C(groups{i}, groups{i}) = C(groups{i}, groups{i}) + diff;
end
for i = 1:l
	C(i, i) = 0.001;
end


function groups = sampleGroups(max_id, sizes)
% Child function, sample overlapped groups
% Input - max_id, max id of nodes, i.e. J or K
%         sizes, a vector indicate the size of each group we want
% Output - groups, cell array in which each cell is a set of node id
n = length(sizes);
groups = cell(n, 1);
for i = 1:n
    groups{i} = randi(max_id, [sizes(i) 1]);
end