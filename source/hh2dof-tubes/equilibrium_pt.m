function [eqPt] = equilibrium_pt(eqNum, parameters)

% Equilibrium_pt solves the equilibrium points for the Henon-Heiles system. 
%       parameters = [mass_a, mass_b, omega_x, omega_y, delta];
% These are numbered as follows.
%-------------------------------------------------------------------------------
%   Indices for the equilibrium points on the potential energy surface:
%
%                               saddle (EQNUM = 2)
% 
% 
%                               center (EQNUM = 1)     
%
% 
%   saddle (EQNUM = 3)                                      saddle (EQNUM = 4)
%-------------------------------------------------------------------------------
%   
    
    % All the equilibrium points are known analytically
    if eqNum == 1 
        eqPt = [0,0,0,0];
    elseif 	eqNum == 3
        xe = -parameters(3)*sqrt( 0.5*parameters(4)^2 ...
            + (parameters(5)/4)*parameters(3)^2);
        eqPt = [xe, -0.5*parameters(3)^2, 0, 0];
    elseif 	eqNum == 4
        xe = parameters(3)*sqrt( 0.5*parameters(4)^2 ...
            + (parameters(5)/4)*parameters(3)^2);
        eqPt = [xe, -0.5*parameters(3)^2, 0, 0];
    elseif 	eqNum == 2
        eqPt = [0, parameters(4)^2/parameters(5), 0, 0];
    end
    
 
end












