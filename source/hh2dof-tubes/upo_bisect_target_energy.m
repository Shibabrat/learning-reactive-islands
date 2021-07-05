function  [x0_PO, T_PO, ePO] = upo_bisect_target_energy(x0po, ...
                                            energyTarget, po_target_file, ...
                                            par)
%
% poTargetEnergy_deleonberne computes the periodic orbit of target energy using
% bisection method. Using bisection method on the lower and higher energy
% values of the POs to find the PO with the target energy. Use this
% condition to integrate with event function of half-period defined by
% maximum distance from the initial point on the PO
% 
% INPUT
% x0po:     Initial conditions for the periodic orbit with the last two
% initial conditions bracketing (lower and higher than) the target energy
% par: [MASS_A MASS_B EPSILON_S D_X LAMBDA ALPHA]
% 
% OUTPUTS
% x0_PO:    Initial condition of the periodic orbit (P.O.)
% T_PO:     Time period 
% ePO:     Energy of the computed P.O.
% 

%     global eSaddle
    global y0 % shared global variable with ODE integration function
%     label_fs = 10; axis_fs = 15; % small fontsize
    label_fs = 20; axis_fs = 30; % fontsize for publications 
    
    iFam = size(x0po,1);

    energyTol = 1e-10;
    tpTol = 1e-6;
    show = 1;   % for plotting the final PO
    
% bisection method begins here
    iter = 0;
    iterMax = 200;
    a = x0po(end-1,:);
    b = x0po(end,:);
    
    fprintf('Bisection method begins \n');
    while iter < iterMax
%         dx = 0.5*(b(1) - a(1));
%         dy = 0.5*(b(2) - a(2));
        
%         c = [a(1) + dx a(2) + dy 0 0];
        c = 0.5*(a + b); % guess based on midpoint
        
        [x0po_iFam,tfpo_iFam] = upo_diff_corr_fam(c, par);
        energyPO = total_energy(x0po_iFam, par);
%         x0po(iFam,1:N) = x0po_iFam ;
%         T(iFam,1)      = 2*tfpo_iFam ;
        
        c = x0po_iFam;
        iter = iter + 1;
        
        if (abs(total_energy(c, par) - energyTarget) < energyTol) || (iter == iterMax)
            fprintf('Initial condition: %e \t %e \t %e \t %e\n', c);
            fprintf('Energy of the initial condition for PO %e\n', ...
                total_energy(c, par));
            x0_PO = c
            T_PO = 2*tfpo_iFam; 
            ePO = total_energy(c, par);
            break
        end
        
        if sign( total_energy(c, par) - energyTarget ) == ...
            sign ( total_energy(a, par) - energyTarget )
            a = c;
        else
            b = c;
        end
        fprintf('Iteration number %d, energy of PO: %f\n',iter, energyPO) ;
        
    end
    fprintf('Iterations completed: %d, error in energy: %e \n', ...
        iter, abs(total_energy(c, par) - energyTarget));
    
%     if abs(t1(end) - tfpo_iFam) < tpTol % happy with convergence of bisection
%         fprintf(['Difference in TP between diff. corr and max. distance ',...
%                 'event %e\n'], abs(t1(end) - tfpo_iFam));
%     end
    
    
    % this is the check if the initial condition is really on a P.0.
    % Integrate with maximum distance event and check if the event time is
    % within tolerance of the half-period: this approach didn't work. 
    % Integrating using the same event as used for differential correction
    RelTol = 3.e-14; AbsTol = 1.e-14; 
%     model = 'barbanis2dof';  
    tspan = [0 20]; % allow sufficient time for the half-period crossing event         
%     OPTIONS = odeset('RelTol',RelTol,'AbsTol',AbsTol,'Events','on'); 
    OPTIONS = odeset('RelTol',RelTol,'AbsTol',AbsTol, ...
        'Events',@half_period_event); 
    
    dum = [c 2*tfpo_iFam energyPO];
    save(po_target_file,'dum','-ascii','-double')
    
%     save('po_traj.txt','po_traj','-ascii','-double');

end





