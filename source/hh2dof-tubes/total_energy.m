function e = total_energy(orbit, parameters)

%   total_energy computes the total energy of an input orbit (represented
%   as M x N with M time steps and N = 4, dimension of phase space for the
%   model) for the 2 DoF solute-solvent model with LJ term in the potential.
% 
%   Orbit can be different initial conditions for the periodic orbit of
%   different energy. When trajectory is input, the output energy is mean.
%

    
    x = orbit(:,1);
    y = orbit(:,2);
    px = orbit(:,3);
    py = orbit(:,4);
    
    e = (px.^2/(2*parameters(1))) + (py.^2/(2*parameters(2))) ...
        + potential_energy(x, y, parameters);
    
    if length(e) > 1 
        e = mean(e);
    end
        
    
end
