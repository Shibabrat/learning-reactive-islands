function pot_energy = potential_energy(x, y, parameters)
    
% parameters = [mass_a, mass_b, omega_x, omega_y, delta];
        
    pot_energy = 0.5*parameters(3)^2*x.^2 + 0.5*parameters(4)^2*y.^2 ...
        + x.^2.*y - (parameters(5)/3)*y.^3;
                
    
end


