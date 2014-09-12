% For choose a good threshold for correlation graph

% Common Settings
N = 50;
K = 50;
J = 50;
is_plot = 1;
num_to_run = 100;


avg_pre1 = zeros(9, 1);
avg_rec1 = zeros(9, 1);
avg_pre2 = zeros(9, 1);
avg_rec2 = zeros(9, 1);
for t = 1:num_to_run
    s1 = 0.5;
    s2 = 0.5;
    group_num = floor(sqrt(K * J / (K + J))) + 1;
    sizes1 = randi(floor(sqrt(K)), 1, group_num);
    sizes2 = randi(floor(sqrt(J)), 1, group_num);
    diff = randi([1 2]);

    % Make data
    [E1 A1 E2 A2 Y X B noise] = simuData(s1, sizes1,...
                                 s2, sizes2, N, K, J, diff);

    % C1, C2
    Y = Y - repmat(mean(Y), N, 1);
    X = X - repmat(mean(X), N, 1);
    C1 = abs(corr(Y) - eye(K));
    C2 = abs(corr(X) - eye(J));

    for i = 1:9
        thres = 0.1 * i;
        C1_thres = C1 > thres;
        C2_thres = C2 > thres;
        [mcc1 TP1 TN1 FP1 FN1] = MCC(A1, C1_thres);
        [mcc2 TP2 TN2 FP2 FN2] = MCC(A2, C2_thres);
        if TP1 ~= 0
            avg_pre1(i) = avg_pre1(i) + TP1 / (TP1 + FP1);
            avg_rec1(i) = avg_rec1(i) + TP1 / (TP1 + FN1);
        end
        if TP2 ~= 0
            avg_pre2(i) = avg_pre2(i) + TP2 / (TP2 + FP2);
            avg_rec2(i) = avg_rec2(i) + TP2 / (TP2 + FN2);
        end
    end
end
avg_pre1(i) = avg_pre1(i) / num_to_run;
avg_pre1(i) = avg_pre1(i) / num_to_run;
avg_pre1(i) = avg_pre1(i) / num_to_run;
avg_pre1(i) = avg_pre1(i) / num_to_run;

if is_plot == 1
    lw = 2;
    plot(0.1:0.1:0.9, avg_pre1, '-ro', 'LineWidth', lw);
    hold on;
    plot(0.1:0.1:0.9, avg_rec1, '-b+', 'LineWidth', lw);
    title('Label Graph');
    hold off;
    figure;
    plot(0.1:0.1:0.9, avg_pre2, '-ro', 'LineWidth', lw);
    hold on;
    plot(0.1:0.1:0.9, avg_rec2, '-b+', 'LineWidth', lw);
    title('Feature Graph');
    hold off;
end