% Data Preparation
load("Data.mat")

format long;

X = [class{1,1}';class{1,2}'];
Y = [ones(120,1);-ones(120,1)];
p = randperm(240);
X = X(p,:);
Y = Y(p);

N = size(X,1);  % number of trials
n = size(X, 2); % number of features

% Hyperparamter Initialization
%   setPara : Initialized parameters
%            setPara.t      
%            setPara.zeta   
%            setPara.Tmax   
%            setPara.tol    
%            setPara.W      
%            setPara.C 

setPara.t = 1;
setPara.C = [0.01; 1; 100; 10000];  % This is called eta in P2
setPara.Tmax = 100;
setPara.tol = 0.001;

w = rand(n, 1);
b = rand(1,1);
W = [w;b];
setPara.W = W;

XI = zeros(N,1);
for i = 1:N
    xi = X(i,:)';
    yi = Y(i);
    XI(i) = max(0, 1-yi*(w'*xi+b)) + 0.001;
end

fnLoss = @loss;

% Cross Validation
num_folds_main = 6;
fold_accuracies_main = zeros(num_folds_main, 1); % This will store the accuracies of each fold
fold_size_main = floor(N / num_folds_main);

for fold = 1:num_folds_main

    start_index = (fold-1) * fold_size_main + 1;
    end_index = fold * fold_size_main;
    X_testing_data = X(start_index:end_index, :);
    X_training_data = [X(1:start_index-1, :); X(end_index+1:end, :)];
    Y_testing_data = Y(start_index:end_index);
    Y_training_data = [Y(1:start_index-1); Y(end_index+1:end)]; 
  
    XI_training_main = [XI(1:start_index-1); XI(end_index+1:end)];
    setPara.zeta = XI_training_main;

    [optSol, optEta] = barrierMethod(X_training_data, Y_training_data, setPara);

    add_column = ones(size(Y_testing_data, 1),1);
    X_testing_data = [X_testing_data, add_column];
    prediction = X_testing_data*optSol;
    for k =1:size(prediction, 1)
        if prediction(k) <0
            prediction(k) = -1;
        else
            prediction(k) = 1;
        end
    end
    correct_classes = prediction == Y_testing_data;
    fold_accuracies_main(fold) = sum(correct_classes)/size(prediction, 1)*100;
    disp(fold_accuracies_main(fold));
end

%accuracy = sum(fold_accuracies_main)/num_folds_main; % accuracy shoudl'nt be this. rather the last accuracy calculated
accuracy = sum(fold_accuracies_main)/6;
disp(accuracy);

optb = optSol(end);
optW = optSol(1:end-1);
absVector = abs(optW);
[sortedValues, sortedIndices] = sort(absVector, 'descend');
top_five_dominant_weights = optW(sortedIndices(1:5));


