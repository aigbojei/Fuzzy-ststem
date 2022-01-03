clc, clear, close all
rng default
global initial_flag

%% particle swarm optimization for 200 iteration

options = optimoptions('particleswarm','PlotFcn', {@pswplotbestf});
%rng default

inital_flag = 0;
for i = 1:200
    %D = 2 for function 1
    [sp_x, ]