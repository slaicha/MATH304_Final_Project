function [optSol, err] = NewtonMethod(fnLoss, theta0, eta, t, X, Y, tol)
% Solve the optimization problem using Newton method
%
% INPUTS:
%   fnLoss: Function handle of F(Z)
%   theta0: Initial value of parameters
%   eta: Hyperparameter in objective function
%   t: parameter of barrier function
%   X: data containing the features
%   Y: data containing the labels
%   tol: Tolerance
%
% OUTPUTS:
%   optSol: Optimal solution
%   err: Error
%
% Your Name: Aicha Slaitane
% Email: as1233@duke.edu
% Date: 2023-10-17

    % Initial values of parameters
    theta = theta0;
    %disp(size(theta));  

    % Backtracking line search parameters. Their values are randomly assigned.
    alpha = 1/3;
    beta = 1/2;

    err = inf;

    iter = 0;
    vals = []; % This list stores the values of the objective function after each iteration
    steps = []; % Stores the values of step sizes after each iteration
 

    while err > tol    % Try with max iter if doesn't converge
        iter = iter + 1;
        [F, G, H] = fnLoss(theta, X, Y, eta, t);
        vals = [vals, F];

        % Compute the Newton Step and Decrement
        % I faced a problem of ill conditionning so I'm trying this
        % H = H + eye(size(H,1));
        % [V, D] = eig(H); 
        % D = diag(D);
        % H_inv = V * D * inv(V);

        nw_step = -H\G;
        decrement = -G'*nw_step;
        
        % Stopping Criterion
        if (decrement)/2 <= tol
            break;
        end

        % Backtracking Line Search: choose step size
        step_size = 1;
        steps = [steps, step_size];
        
        temp_theta = theta + step_size * nw_step;
        [F_step, ~, ~] = fnLoss(temp_theta, X, Y, eta, t);
        
        while F_step < F + alpha * step_size * G' * nw_step
            step_size = beta * step_size;
            temp_theta = theta + step_size * nw_step;
            [F_step, ~, ~] = fnLoss(temp_theta, X, Y, eta, t);

            steps = [steps, step_size];
            vals = [vals, F_step];
        end
        % Update theta
        theta = temp_theta;
        
        err = abs(F_step - F);
     end


    %optVal = vals(length(vals));  % Final value of the objective function; is optSol theta of the function value
    optSol = theta;

    % Try to visualize the results (values of the objective function vals) with regards to step size (steps) and iter (iter). 

end





