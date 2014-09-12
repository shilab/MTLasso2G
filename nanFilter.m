function C = nanFilter(C)
% filter NaN values in correlation matrix by setting them to zero

C(isnan(C)) = 0;