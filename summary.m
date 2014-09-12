ID = 2:6:60;
ID1 = ID(1:5);
ID2 = ID(6:10);

AUCs1 = meanAUCs1(ID1, :);
AUCs2 = meanAUCs1(ID2, :);
Errors1 = meanErrors1(ID1, :);
Errors2 = meanErrors1(ID2, :);
stdAUCs1 = sdAUCs1(ID1, :);
stdAUCs2 = sdAUCs1(ID2, :);
stdErrors1 = sdErrors1(ID1, :);
stdErrors2 = sdErrors1(ID2, :);

% Plot AUCs
errorbar(100:100:500, AUCs1(:, 1), stdAUCs1(:, 1)/2, '-b', 'LineWidth', 2);
hold on;
errorbar(100:100:500, AUCs1(:, 2), stdAUCs1(:, 2)/2, '--g', 'LineWidth', 2);
errorbar(100:100:500, AUCs1(:, 3), stdAUCs1(:, 3)/2, '--rs', 'LineWidth', 2);
errorbar(100:100:500, AUCs1(:, 4), stdAUCs1(:, 4)/2, '-.c', 'LineWidth', 2);
errorbar(100:100:500, AUCs1(:, 5), stdAUCs1(:, 5)/2, '-*m', 'LineWidth', 2);
legend('Lasso','MTLasso','MTLasso LG', 'MTLasso FG', 'MTLasso 2G');
xlabel('J');
ylabel('AREA UNDER PR-CURVE');
title('K=10');
axis([100-1 500+1 0.65 1]);

figure;
errorbar(100:100:500, AUCs2(:, 1), stdAUCs2(:, 1)/2, '-b', 'LineWidth', 2);
hold on;
errorbar(100:100:500, AUCs2(:, 2), stdAUCs2(:, 2)/2, '--g', 'LineWidth', 2);
errorbar(100:100:500, AUCs2(:, 3), stdAUCs2(:, 3)/2, '--rs', 'LineWidth', 2);
errorbar(100:100:500, AUCs2(:, 4), stdAUCs2(:, 4)/2, '-.c', 'LineWidth', 2);
errorbar(100:100:500, AUCs2(:, 5), stdAUCs2(:, 5)/2, '-*m', 'LineWidth', 2);
legend('Lasso','MTLasso','MTLasso LG', 'MTLasso FG', 'MTLasso 2G');
xlabel('J');
ylabel('AREA UNDER PR-CURVE');
title('K=50');
axis([100-1 500+1 0.65 1]);

% Plot Errors
figure;
errorbar(100:100:500, Errors1(:, 1), stdErrors1(:, 1)/2, '-b', 'LineWidth', 2);
hold on;
%errorbar(100:100:500, Errors1(:, 2), stdErrors1(:, 2)/2, '--g', 'LineWidth', 2);
errorbar(100:100:500, Errors1(:, 3), stdErrors1(:, 3)/2, '--rs', 'LineWidth', 2);
errorbar(100:100:500, Errors1(:, 4), stdErrors1(:, 4)/2, '-.c', 'LineWidth', 2);
errorbar(100:100:500, Errors1(:, 5), stdErrors1(:, 5)/2, '-*m', 'LineWidth', 2);
legend('Lasso', 'MTLasso LG', 'MTLasso FG', 'MTLasso 2G');
xlabel('J');
ylabel('L^1 ERROR');
title('K=10');
axis([100-1 500+1 0 80]);

figure;
errorbar(100:100:500, Errors2(:, 1), stdErrors2(:, 1)/2, '-b', 'LineWidth', 2);
hold on;
%errorbar(100:100:500, Errors2(:, 2), stdErrors2(:, 2)/2, '--g', 'LineWidth', 2);
errorbar(100:100:500, Errors2(:, 3), stdErrors2(:, 3)/2, '--rs', 'LineWidth', 2);
errorbar(100:100:500, Errors2(:, 4), stdErrors2(:, 4)/2, '-.c', 'LineWidth', 2);
errorbar(100:100:500, Errors2(:, 5), stdErrors2(:, 5)/2, '-*m', 'LineWidth', 2);
legend('Lasso','MTLasso LG', 'MTLasso FG', 'MTLasso 2G');
xlabel('J');
ylabel('L^1 ERROR');
title('K=50');
axis([100-1 500+1 0 400]);