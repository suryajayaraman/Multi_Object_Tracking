%% test RMSE class functions
clear all; clc; close all;

N = 20;
x = cell(N,1);
y = cell(N,1);

for i = 1:N
    x{i} = randn(4,1);
    y{i} = randn(4,1);
end

root_mean_square_error = RMSE(x,y);