% for sge scripts
% Author: Xing Xu @ TTIC
function run(N, K, J, ID, num_to_run, is_corr, weight_func)
num_fold = 3;
max_it = 20;
[corr_thres1 corr_thres2] = getCorrThreshold(K, J);

% Re-generate random seed according to system clock
reset(RandStream.getDefaultStream,sum(100*clock));

% run and save
res = cell(num_to_run, 1);
for i = 1:num_to_run
	res{i} = runMTLassos(N, K, J, weight_func, num_fold, max_it, 0, is_corr, corr_thres1, corr_thres2);
end
clear i;
save(sprintf('result_%d_%d_%d_%d', N, K, J, ID));
