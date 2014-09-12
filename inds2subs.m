function subs = inds2subs(inds, sizes)

num = length(inds);
subs = zeros(2, num);
for i = 1:num
    [sub1 sub2] = ind2sub(sizes, inds(i));
    subs(:, i) = [sub1; sub2];
end