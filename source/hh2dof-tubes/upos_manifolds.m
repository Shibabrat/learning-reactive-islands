%--------------------------------------------------------------------------
%   Indices for the equilibrium points on the potential energy surface:
%
%                          saddle (EQNUM = 2)
% 
% 
%                          center (EQNUM = 1)     
%
% 
%   saddle (EQNUM = 3)                               saddle (EQNUM = 4)
%--------------------------------------------------------------------------
global eqNum deltaE

% Setting up parameters and global variables
N = 4;          % dimension of phase space
parameters = [1, 1, 1, 1, 1];

eqNum = 2;  
[eqPt] = equilibrium_pt(eqNum, parameters);

% energy of the saddle equilibrium point
eSaddle = total_energy(eqPt, parameters);

n_mfd_traj = 25;
% n_mfd_traj = 5000;
target_energy = 0.19;
deltaE = target_energy - eSaddle;

% tmfd = 6*TPOFam(nMed);
tmfd = 30;

%%

nFam = 100; % use nFam = 10 for low energy

% first two amplitudes for continuation procedure to get p.o. family
% Ax1  = 2.e-5; % initial amplitude (1 of 2) values to use: 2.e-3
% Ax2  = 2*Ax1; % initial amplitude (2 of 2)

% amplitudes for mu_2 = 0.1
% Ax1 = 1.e-8;
% Ax2 = 2*Ax1;

Ax1 = 1.e-5;
Ax2 = 2*Ax1;

tic;
%  get the initial conditions and periods for a family of periodic orbits
po_fam_file = ['x0_tp_fam_eqPt',num2str(eqNum),'.txt'];
[po_x0Fam,po_tpFam] = upo_family(eqNum, Ax1, Ax2, nFam, po_fam_file, parameters) ; 

poFamRuntime = toc;

x0podata = [po_x0Fam, po_tpFam];


%%

% paramstarttime = tic;    
po_fam_file = ['x0_tp_fam_eqPt', num2str(eqNum),'.txt'];
            
fprintf('Loading the periodic orbit family from data file %s \n',po_fam_file); 

x0podata = importdata(po_fam_file);

po_brac_file = ['x0po_T_energyPO_eqPt',num2str(eqNum), ...
                '_brac',num2str(deltaE),'.txt'];

tic;
[x0poTarget,TTarget] = upo_bracket_target_energy(target_energy, ...
                        x0podata, po_brac_file, parameters);
poTarE_runtime = toc;

save(['model_parameters_eqPt',num2str(eqNum), ...
        '_E',num2str(deltaE), '.txt'], 'parameters', '-ASCII', '-double');


%% %

% target specific periodic orbit
% Target PO of specific energy with high precision; does not work for the
% model 

po_target_file = ['x0po_T_energyPO_eqPt',num2str(eqNum), ...
                    '_DelE',num2str(deltaE),'.txt'];

[x0_PO, T_PO, e_PO] = upo_bisect_target_energy(x0poTarget, ...
                        target_energy,po_target_file,parameters);


data_path = ['./x0po_T_energyPO_eqPt', num2str(eqNum), ...
            '_DelE', num2str(deltaE), '.txt'];




%%

frac = 0;
del = 1e-8;

x0po = importdata(data_path);

TPOFam = x0po(:,5); 
ePOFam = x0po(:,6);
nMed = size(x0po,1);

  

stbl = -1;
if eqNum == 4 
    dir = -1;   % inside the well
else
    dir = 1;    % inside the well
end
[xW,x0W] = upo_mani_globalize(x0po(nMed,1:4), TPOFam(nMed), frac, stbl, dir, ...
                            del, tmfd, n_mfd_traj, parameters);


energyTube = ePOFam(nMed) ;
title(['Total energy: ', num2str(energyTube)]);

hold on


%% %

% tmfd = 7*TPOFam(nMed);
% stbl = 1;
% dir = 1; 
% [xW,x0W] = upo_mani_globalize(x0po(nMed,1:4),TPOFam(nMed), frac, stbl, dir, ...
%                             del,tmfd,n_mfd_traj,parameters);
% 
% 
% energyTube = ePOFam(nMed) ;
% title(['Total energy: ', num2str(energyTube)]);
% 
% paramendtime = toc;



%%



