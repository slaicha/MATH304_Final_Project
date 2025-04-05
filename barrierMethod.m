function [optSol, optEta] = barrierMethod(X, Y, setPara)
% Get the optimal solution using interior point algorithm and get the
% optimal eta using five fold cross-validation from the given eta set
%
% INPUTS:   
%   X(MxN) : trData(i,j) is the i-th feature from the j-th trial
%   Y(Nx1): trData(j) is the label of the j-th trial (1 or -1)
%   setPara : Initialized parameters
%            setPara.t      
%            setPara.zeta   
%            setPara.Tmax   
%            setPara.tol    
%            setPara.W      
%            setPara.C      
%
% OUTPUTS:
%   optiEta: Optimal eta value 
%   optSol: Optimal solution     
%
% Your Name: Aicha Slaitane
% Email: as1233@duke.edu
% Date: 2023-10-17

% You can call your NewtonMethod as follows:
% [optSol, err] = solver_Newton(@loss, theta0, eta, t, X, Y, tol);

    % Initialized parameters
    t = setPara.t;          % Barrier method parameter
    XI = setPara.zeta;      % Slack variables?
    Tmax = setPara.Tmax;    % Max number of iterations
    tol = setPara.tol;      % Tolerance value for stopping Newton's method
    W = setPara.W;          % [optSol, err] = NewtonMethod(fnLoss, theta0, eta, t, X, Y, tol)?
    eta_range = setPara.C;        % eta on P2. Regularization parameter. I guess this is the range [0.01; 1; 100; 10000]; eta_range should be column vector

    N = size(X,1);
    n = size(X,2);
    mu = 15;                % Barrier update factor
    m = 2*N;                % Number of inequality contraints
    iter = 0;
    
    N = size(X,1);
    num_folds = 5;
    eta_accuracy = zeros(size(eta_range, 1), 1);
    fold_size = floor(N / num_folds); 
    opt_solutions = []; % stores the optimal solutions theta (weights and biases) corresponding to each eta

    for i = 1:size(eta_range, 1)
        
        eta = eta_range(i);
        accuracy = zeros(num_folds,1); % accuracies of the validation sets of each eta
        temp_t = t;
        temp_W = W;
        %disp(eta);

        for fold = 1:num_folds

            start_index = (fold-1) * fold_size + 1;
            end_index = fold * fold_size;
            X_validation_data = X(start_index:end_index, :);
            Y_validation_data = Y(start_index:end_index);
            X_training_data = [X(1:start_index-1, :); X(end_index+1:end, :)];
            Y_training_data = [Y(1:start_index-1); Y(end_index+1:end)];
           
            XI_training = [XI(1:start_index-1); XI(end_index+1:end)];
            temp_theta_training = [W; XI_training];
            
            % training; barrier's method
            while iter < Tmax
                iter = iter + 1;
                % Centering Step using Newton's method
                [new_theta, ~] = NewtonMethod(@loss, temp_theta_training, eta, temp_t, X_training_data, Y_training_data, tol);   
                % Check for convergence
                if (m / temp_t) <= tol
                    break;
                end
                % Update t
                temp_t = temp_t * mu;
                temp_theta_training = new_theta;
            end

            iter = 0;
            temp_t = t;

            % calculating the accuracy of the specific validation fold
            temp_W = new_theta(1:n+1); % contains both the weights and bias
            
            add_column = ones(size(Y_validation_data, 1),1);
            X_validation_data = [X_validation_data, add_column];
            prediction = X_validation_data*temp_W;
            for k =1:size(prediction, 1)
                if prediction(k) <0
                    prediction(k) = -1;
                else
                    prediction(k) = 1;
                end
            end
            %disp(prediction);
            correct_classes = prediction == Y_validation_data;
            %disp(sum(correct_classes));
            accuracy(fold) = sum(correct_classes)/size(prediction, 1)*100;
            %disp(accuracy(fold));
        end
        eta_accuracy(i) = sum(accuracy)/num_folds;
        opt_solutions = [opt_solutions; temp_W];
    end
   
    %disp(eta_accuracy);
    [~, max_idx] = max(eta_accuracy);
    optEta = eta_range(max_idx);
    optSol = opt_solutions((n+1)*(max_idx-1)+1:(n+1)*max_idx);

end

