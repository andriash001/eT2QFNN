%% THIS IS A MATLAB MFILE TO RUN THE eT2QFNN ALGORITHM
clc
clear
close all

%% LOAD THE DATA
load databj.mat                 %The matrix format is as follows: data = [X Y]
N = length(data);               %number of observation
training_data = round(0.7*N);   %the number of training data, 70%.
I = 2;                          %number of input

%% DETERMINE THE HYPERPARAMETER
lr = 0.01;                      %learning rate

%% RUN THE ALGORITHM
mode = 3;                       %1: BINARY classification, 3: REGRESSION
[y,rule,network,time] = eT2QFNN(data,I,lr,mode,training_data);
