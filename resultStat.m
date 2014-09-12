function [meanAUC sdAUC meanError sdError] = resultStat(res)
% Summarize the results for one setup
num = 50;

%% Average and median AUC
areas = zeros(num, 5);
for i = 1:num
    areas(i, :) = roc(res{i}{1}, res{i}{2}, res{i}{3}, res{i}{4}, res{i}{5}, res{i}{6}, 0);
end
meanAUC = mean(areas);
sdAUC = std(areas);


%% Average L1 error
errors = zeros(num, 5);
for i = 1:num
    errors(i, :) = [sum(sum(abs(res{i}{1} - res{i}{2}))) sum(sum(abs(res{i}{1} - res{i}{3})))...
        sum(sum(abs(res{i}{1} - res{i}{4}))) sum(sum(abs(res{i}{1} - res{i}{5})))...
        sum(sum(abs(res{i}{1} - res{i}{6})))];
end
meanError = mean(errors);
sdError = std(errors);