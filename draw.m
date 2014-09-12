function draw(A)
% draw network using graphviz
addpath(genpath('/usr/local/bin/graphViz4Matlab'));
graphViz4Matlab('-adjMat', A, '-undirected', true);