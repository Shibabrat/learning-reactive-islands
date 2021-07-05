function [xDot,isterminal,direction] = unstable_intersect_sosaty0(t,x) 

% Function to catch event of crossing y = 0 section
    
    xDot = x(2);
    % The value that we want to be zero    
    if abs(t) > 1e-2 
        isterminal = 1; % Halt integration 
    else
        isterminal = 0; % don't terminate within a short time
    end
    
    % The zero can be approached from either direction
    direction = 1; % unstable manifold's 

end