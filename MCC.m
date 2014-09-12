% Function calculate Matthews correlation coefficient
% as a criterion for matrix recovery
% Author: Xing Xu @ TTIC
% Last Update: 2011-9-14

function [res TP TN FP FN] = MCC(B, B_hat)
% Input - B, the true matrix with 0/1 entries
%         B_hat, the predicted matrix, entries with non-zero value will
%                be assigned 1
% Output - MCC value between prediction and real value
B(B ~= 0) = 1;
B_hat(B_hat ~= 0) = 1;
TP = length(find(B + B_hat == 2));
TN = length(find(B + B_hat == 0));
FP = length(find(B - B_hat == -1));
FN = length(find(B - B_hat == 1));

res = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));