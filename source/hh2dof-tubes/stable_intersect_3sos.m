function [xDot,isterminal,direction] = stable_intersect_3sos(t,x) 

    % Function to catch event of crossing y = 0.5, 0, -0.25 sections and
    % terminate when crossing abs(x) = 2

    xDot = [x(2) - 0.5; x(2); x(2) + 0.25; abs(x(1)) - 2];
    % The value that we want to be zero
    if abs(t) > 1e-2 
        isterminal = [0; 0; 0; 1]; % Halt integration
    else
        isterminal = [0; 0; 0; 1]; % don't terminate within a short time
    end
%     isterminal = 0;

    % The zero can be approached from either direction
    direction = [-1; 0]; % stable manifold's intersection with py > 0
%     direction = 1; % stable manifold's intersection with py < 0

end