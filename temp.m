% for i = 1:7
%     for j = 1:7
%         res(i, j) = sum(sum(abs(B{i} - B{j})));
%         fprintf('%d\t', res(i, j));
%     end
%     fprintf('\n');
% end

for i = 1:7
    for j = 1:7
        res(i, j) = MCC(B{i}, B{j});
        fprintf('%.3f\t', res(i, j));
    end
    fprintf('\n');
end