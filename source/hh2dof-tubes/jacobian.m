function Df = jacobian(eqPt, parameters)
   
    % parameters = [mass_a, mass_b, omega_x, omega_y, delta];
    
    x = eqPt(:,1);
    y = eqPt(:,2);
    px = eqPt(:,3);
    py = eqPt(:,4);
    
    Df = zeros(4,4);

        
    deriv_wrt_x_f3 = - parameters(3)^2 - 2*y;
    
    deriv_wrt_y_f3 = - 2*x;
    
    deriv_wrt_x_f4 = - 2*x;
    
    deriv_wrt_y_f4 = - parameters(4)^2 + 2*parameters(5)*y; 

    Df(1,3) = 1/parameters(1);
    Df(2,4) = 1/parameters(2);
    
    Df(3,1) = deriv_wrt_x_f3;

    Df(3,2) = deriv_wrt_y_f3;
        
    Df(4,1) = deriv_wrt_x_f4;
    
    Df(4,2) = deriv_wrt_y_f4;

    
    
end




