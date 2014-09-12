% Plot MCCs for corr graph and abs weight
folder_path = 'results/1/';
what_path = what(folder_path);
filenames = what_path.mat;
num = length(filenames);

MCCs = cell(num, 1);
for i = 1:num
    load([folder_path filenames{i}]);
    thres = getThreshold(N, K, J);
    for j = 1:50
        B_prime1 = res{j}{2};
        B_prime1(abs(B_prime1) < thres) = 0;
        MCCs{i}(j, 1) = MCC(res{j}{1}, B_prime1);
        B_prime2 = res{j}{3};
        B_prime2(abs(B_prime2) < thres) = 0;
        MCCs{i}(j, 2) = MCC(res{j}{1}, B_prime2);
        B_prime3 = res{j}{4};
        B_prime3(abs(B_prime3) < thres) = 0;
        MCCs{i}(j, 3) = MCC(res{j}{1}, B_prime3);
        B_prime4 = res{j}{5};
        B_prime4(abs(B_prime4) < thres) = 0;
        MCCs{i}(j, 4) = MCC(res{j}{1}, B_prime4);
        B_prime5 = res{j}{6};
        B_prime5(abs(B_prime5) < thres) = 0;
        MCCs{i}(j, 5) = MCC(res{j}{1}, B_prime5);
        
        if MCCs{i}(j, 5) > MCCs{i}(j, 4) && MCCs{i}(j, 5) > MCCs{i}(j, 3) &&...
                MCCs{i}(j, 3) > MCCs{i}(j, 2) && MCCs{i}(j, 3) > MCCs{i}(j, 1) && 0
            figure;
            subplot(2, 3, 1);
            colorspy(res{j}{1});
            title('True B');
            subplot(2, 3, 2);
            colorspy(B_prime1);
            title('Lasso Estimated');
            subplot(2, 3, 3);
            colorspy(B_prime2);
            title('MTLasso Estimated');
            subplot(2, 3, 4);
            colorspy(B_prime3);
            title('MTLasso LG Estimated');
            subplot(2, 3, 5);
            colorspy(B_prime4);
            title('MTLasso FG Estimated');
            subplot(2, 3, 6);
            colorspy(B_prime5);
            title('MTLasso 2G Estimated');
        end
    end
end

ID = 2:2:20;
ID1 = ID(1:5);
ID2 = ID(6:10);
MCCs1 = MCCs(ID1);
MCCs2 = MCCs(ID2);

% K = 10 and K = 50
meanMCCs1 = zeros(5, 5);
stdMCCs1 = zeros(5, 5);
meanMCCs2 = zeros(5, 5);
stdMCCs2 = zeros(5, 5);
for i = 1:5
    meanMCCs1(i, :) = nanmean(MCCs1{i});
    stdMCCs1(i, :) = nanstd(MCCs1{i});
    meanMCCs2(i, :) = nanmean(MCCs2{i});
    stdMCCs2(i, :) = nanstd(MCCs2{i});
end

% Plot
errorbar(100:100:500, meanMCCs1(:, 1), stdMCCs1(:, 1)/2, '-b', 'LineWidth', 2);
hold on;
errorbar(100:100:500, meanMCCs1(:, 2), stdMCCs1(:, 2)/2, '--g', 'LineWidth', 2);
errorbar(100:100:500, meanMCCs1(:, 3), stdMCCs1(:, 3)/2, '--rs', 'LineWidth', 2);
errorbar(100:100:500, meanMCCs1(:, 4), stdMCCs1(:, 4)/2, '-.c', 'LineWidth', 2);
errorbar(100:100:500, meanMCCs1(:, 5), stdMCCs1(:, 5)/2, '-*m', 'LineWidth', 2);
legend('Lasso','MTLasso','MTLasso LG', 'MTLasso FG', 'MTLasso 2G');
xlabel('J');
ylabel('MCC');
title('K=10');
axis([100-1 500+1 0 1]);

figure;
errorbar(100:100:500, meanMCCs2(:, 1), stdMCCs2(:, 1)/2, '-b', 'LineWidth', 2);
hold on;
errorbar(100:100:500, meanMCCs2(:, 2), stdMCCs2(:, 2)/2, '--g', 'LineWidth', 2);
errorbar(100:100:500, meanMCCs2(:, 3), stdMCCs2(:, 3)/2, '--rs', 'LineWidth', 2);
errorbar(100:100:500, meanMCCs2(:, 4), stdMCCs2(:, 4)/2, '-.c', 'LineWidth', 2);
errorbar(100:100:500, meanMCCs2(:, 5), stdMCCs2(:, 5)/2, '-*m', 'LineWidth', 2);
legend('Lasso','MTLasso','MTLasso LG', 'MTLasso FG', 'MTLasso 2G');
xlabel('J');
ylabel('MCC');
title('K=50');
axis([100-1 500+1 0 1]);