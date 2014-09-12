function subs = matrix2assoc(A)
pos = find(A~=0);
subs = [];
for i = 1:length(pos)
    [sub1 sub2] = ind2sub(size(A), pos(i));
    subs = [subs; [sub1 sub2]];
end