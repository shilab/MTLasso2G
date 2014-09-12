function [thres1 thres2] = getCorrThreshold(K, J)

if K < J
    thres1 = 0.4;
    thres2 = 0.6;
else
    thres1 = 0.6;
    thres2 = 0.5;
end