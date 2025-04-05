function [F, G, H] = loss(theta, X, Y, eta, t)
% Compute the cost function F(theta)
%
% INPUTS: 
%   theta: Parameter values
%   X: Features
%   Y: Labels
%   eta and t: hyper-parameter in the objective function
% OUTPUTS
%   F: Function value
%   G: Gradient value
%   H: Hessian value
%
% Your Name: Aicha Slaitane
% Email: as1233@duke.edu
% Date: 2023-10-17

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To improve the execution speed, please implement with matrix
% computation. It is much faster than code using for-loop.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Parameter values in theta. We assume they are all in one column.
    N = size(X, 1);      % Number of trials
    n = size(X, 2);      % Number of features

    w = theta(1:n);
    b = theta(n+1:n+1);
    XI = theta(end-N:end);
    
    lambda = 1/eta;
    
    % First derivatives initialization
    G_w = zeros(n, 1);
    G_b = 0;
    G_XI = zeros(N,1);

    % Second derivatives initialization
    H_ww = zeros(n,n);
    H_wb = zeros(n,1);
    H_wXi = zeros(n,N);
    H_bw = zeros(1,n);
    H_bb = 0;
    H_bXi = zeros(1,N);
    H_Xiw = zeros(N,n);
    H_Xib = zeros(N,1);
    H_XiXi = eye(N);


    val_log = 0; % value to be used in the cost function

    for i = 1:N
        xi = X(i,:)';  % Transpose xi so that it is a column vector
        yi = Y(i);   
        Xi = XI(i);
        
        denominator = w'*xi*yi+b*yi-1+Xi; % To avoid recalculation;
        %disp(denominator);
        xiyi = xi*yi;

        % First derivatives
        G_w = G_w + xiyi/denominator;
        G_b = G_b + yi/denominator;
        G_XI(i) = 1 - 1/t * (1/denominator + 1/Xi);

        % Second derivatives
        H_ww = H_ww + xiyi*xiyi'/(denominator^2);
        %H_wb = H_wb + xiyi * yi/(denominator^2);
        H_wXi(:,i) = xiyi/(denominator^2) * 1/t;
        %H_bw = H_bw + yi * xiyi'/(denominator^2);
        H_bb = H_bb + (yi^2)/(denominator^2);
        %H_bXi(:,i) = yi/(denominator^2) * 1/t;
        %H_Xiw(i,:) = xiyi'/(denominator^2) * 1/t;
        %H_Xib(i,:) = yi/(denominator^2)*1/t;
        H_XiXi(i,i) = 1/t * (1/(denominator^2)+1/(Xi^2));
        

        val_log = val_log + log(denominator*Xi); 
        
    end

    G_w =  2*lambda*w - 1/t * G_w;
    G_b = -1/t * G_b;
    H_ww = 2*lambda*eye(n) + 1/t * H_ww;
    %H_wb = 1/t*H_wb;
    %H_bw = 1/t * H_bw;
    H_bb = 1/t * H_bb;
    %disp(H_ww);

    % Output gradient and Hessian values of the parameters.
    G = [G_w; G_b; G_XI];
    H = [H_ww, H_wb, H_wXi; H_bw, H_bb, H_bXi; H_Xiw, H_Xib, H_XiXi];

    % I will fill add to the diagonal of H 10 because I faced problems
    % with ill conditionnning when calculating the inverse. I want to make
    % sure the diagonals are non zero
    for i = 1:(n+1+N)
       H(i, i) = H(i, i) + 10;
    end
    % This will affect the accuracy but would allow me to run the code
    
    % Objective function value
    F = sum(XI) + lambda * (w'*w) - 1/t * val_log;
     
end

