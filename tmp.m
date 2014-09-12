start = 6;

a = meanAUCs2;
for i = start:6:60
    fprintf('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n', a(i, 1), a(i, 2), a(i, 3), a(i, 4), a(i, 5));
end
fprintf('\n\n');

a = sdAUCs2;
for i = start:6:60
    fprintf('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n', a(i, 1), a(i, 2), a(i, 3), a(i, 4), a(i, 5));
end
fprintf('\n\n');

a = meanErrors2;
for i = start:6:60
    fprintf('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n', a(i, 1), a(i, 2), a(i, 3), a(i, 4), a(i, 5));
end
fprintf('\n\n');

a = sdErrors2;
for i = start:6:60
    fprintf('%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n', a(i, 1), a(i, 2), a(i, 3), a(i, 4), a(i, 5));
end

clear i a start;