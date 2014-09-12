% Summarize the results for all setups in one folder
function [meanAUCs sdAUCs meanErrors sdErrors] = resultsStat(folder_path)
what_path = what(folder_path);
filenames = what_path.mat;
num = length(filenames);

meanAUCs = zeros(num, 5);
sdAUCs = zeros(num, 5);
meanErrors = zeros(num, 5);
sdErrors = zeros(num, 5);
for i = 1:num
    load([folder_path filenames{i}]);
    [meanAUCs(i, :) sdAUCs(i, :) meanErrors(i, :) sdErrors(i, :)] = resultStat(res);
end

for i = 1:num
    fprintf('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n', meanAUCs(i, 1),...
        meanAUCs(i, 2), meanAUCs(i, 3), meanAUCs(i, 4), meanAUCs(i, 5));
end

for i = 1:num
    fprintf('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n', meanErrors(i, 1),...
        meanErrors(i, 2), meanErrors(i, 3), meanErrors(i, 4), meanErrors(i, 5));
end