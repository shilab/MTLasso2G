% Plot the roc curve for Lassos and calculate the AUC
% Author: Xing Xu @ TTIC
% Last Update: 2011-10-6

function area = roc(B, Bhat_0, Bhat_MT, Bhat_1, Bhat_1_prime, Bhat_2, is_plot)

% Settings
lw = 3; ms = 6;
[pre_0 rec_0] = getPreRec(B, Bhat_0);
[pre_MT rec_MT] = getPreRec(B, Bhat_MT);
[pre_1 rec_1] = getPreRec(B, Bhat_1);
[pre_1_prime rec_1_prime] = getPreRec(B, Bhat_1_prime);
[pre_2 rec_2] = getPreRec(B, Bhat_2);

% Plot
if is_plot == 1
    figure;
    plot(pre_0, rec_0, '-b', 'LineWidth', lw, 'MarkerSize', ms);
    hold on;
    plot(pre_MT, rec_MT, '--g', 'LineWidth', lw, 'MarkerSize', ms);
    plot(pre_1, rec_1, ':r', 'LineWidth', lw, 'MarkerSize', ms);
    plot(pre_1_prime, rec_1_prime, '-.c', 'LineWidth', lw, 'MarkerSize', ms);
    plot(pre_2, rec_2, '-.m', 'LineWidth', lw, 'MarkerSize', ms);
    title('PR CURVE');
    xlabel('PRECISION RATIO'); ylabel('RECALL RATIO'); axis([-0.001 1.001 -0.001 1.001]);
    legend('Lasso', 'MTLasso', 'MTLasso LG', 'MTLasso FG', 'MTLasso 2G', 'Location','SouthWest');
    hold off;
    maximize(gcf);
end

% Calculate AUC and print out
area = zeros(5, 1);
area(1) = getArea(pre_0, rec_0);
area(2) = getArea(pre_MT, rec_MT);
area(3) = getArea(pre_1, rec_1);
area(4) = getArea(pre_1_prime, rec_1_prime);
area(5) = getArea(pre_2, rec_2);
fprintf('AUC:%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n', area(1), area(2), area(3), area(4), area(5));


function [pre rec] = getPreRec(B, Bhat)
matri = abs(Bhat);
matri_sort = sort(unique(matri(:)));
thres = matri_sort([1:40:0.9*length(matri_sort) 0.9*length(matri_sort):1:length(matri_sort)]);
num = length(thres);
pre = zeros(num, 1);
rec = zeros(num, 1);

for i = 1:num
    threshold = thres(i);
    Bhat_threshed = Bhat;
    Bhat_threshed(abs(Bhat) < threshold) = 0;
    [mcc TP TN FP FN] = MCC(B, Bhat_threshed);
    pre(i) = TP / (TP + FP);
    rec(i) = TP / (TP + FN);
end

pre = [0; pre; 1];
rec = [1; rec; 0];


function res = getArea(pre, rec)
res = sum((pre(2:end) - pre(1:end-1)) .* ((rec(1:end-1) - rec(2:end)) / 2 + rec(2:end)));