function [x0poGuess,TGuess] = upo_guess_linear(Ax, eqNum, parameters)

%   [x0poGuess,TGuess] = upo_guess_linear(eqNum,Ax,parameters) ;
% 
% Uses a small displacement from the equilibrium point (in a direction 
% on the collinear point's center manifold) as a first guess for a planar 
% periodic orbit (called a Lyapunov orbit in th rest. three-body problem).
%
% The initial condition and period are to be used as a first guess for
% a differential correction routine.
%
% Output:
% x0poGuess = initial state on the periodic orbit (first guess)
%           = [ x 0  0 yvel]  , (i.e., perp. to x-axis and in the plane)
% TGuess    = period of periodic orbit (first guess)
%
% Input:
% eqNum = the number of the equilibrium point of interest
% Ax    = nondim. amplitude of periodic orbit (<< 1) 
%
% 
% 

    x0poGuess  = zeros(4,1);
    
    % Equilibrium point number = 2,3,4 is saddle
    equil_pt = equilibrium_pt(eqNum, parameters);
%     eqPt = [eqPos' 0 0]; % phase space location of equil. point

    % Get the eigenvalues and eigenvectors of Jacobian of ODEs at equil. point
    [Es,Eu,Ec,Vs,Vu,Vc] = eigvalvecs_equil_pt(equil_pt, parameters);

    
    l = abs(imag(Ec(1)))
    
%     Df = jacobian(equil_pt, parameters);
%     k2 = -(parameters(1)*l^2 + Df(3,1))/(Df(3,2));
%     k2 = - Df(4,1)/(parameters(2)*l^2 + Df(4,2));

    k2 = (2*equil_pt(1))/( -parameters(4)^2 ...
        + 2*parameters(5)*equil_pt(2) + l^2*parameters(2) );


    % This is where the linearized guess based on center manifold needs
    % to be entered.
    x0poGuess(1)	= equil_pt(1) + Ax ;
    x0poGuess(2)	= equil_pt(2) + Ax*k2 ;

    
    TGuess = 2*pi/l ;
    
   
end




