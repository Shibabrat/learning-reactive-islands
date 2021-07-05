function PHIdot = variational_equations(t, PHI, parameters)

%        PHIdot = variational_equations(t, PHI, parameters) ;
%
% This here is a preliminary state transition, PHI(t,t0),
% matrix equation based on...
%
%        d PHI(t, t0)
%        ------------ =  Df(t) * PHI(t, t0)
%             dt


    x(1:4) = PHI(17:20);
    phi  = reshape(PHI(1:16),4,4);


    Df = jacobian(x, parameters);
    
    phidot = Df * phi; % variational equation

    PHIdot        = zeros(20,1);
    PHIdot(1:16)  = reshape(phidot,16,1); 
    PHIdot(17:20) = henonheiles(t, x, parameters);
    
%     PHIdot(17)    = x(3)/par(1);
%     PHIdot(18)    = x(4)/par(2);
%     PHIdot(19)    = -dVdx; 
%     PHIdot(20)    = -dVdy;

end