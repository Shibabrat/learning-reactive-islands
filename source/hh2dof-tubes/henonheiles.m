function xDot = henonheiles(t, x, parameters)

% parameters = [mass_a, mass_b, omega_x, omega_y, delta];
    
    xDot = zeros(length(x),1);
   
    xDot(1) = x(3)/parameters(1);

    xDot(2) = x(4)/parameters(2);
    
    xDot(3) = - ( parameters(3)^2*x(1) + 2*x(1).*x(2) );
    
    xDot(4) = - ( parameters(4)^2*x(2) + x(1).^2 - parameters(5)*x(2).^2 );

end