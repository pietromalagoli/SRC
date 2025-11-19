clear all; close all;
addpath(genpath('../dlt_cnn_map_dropout_nobiasnn'));
addpath(genpath('../sleep'));
addpath(genpath('../utils'));

nn = nnsetup([100 4]);
disp(nn.size);
rand(nn.size(2), nn.size(2 - 1))