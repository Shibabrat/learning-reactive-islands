#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate training data for Henon-Heiles Hamiltonian on a SOS with y = y_constant 
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import henonheiles
import importlib
importlib.reload(henonheiles)
import henonheiles as HH2dof

## Setting the parameters
total_energy = 0.20
# m_x, m_y, omega_x, omega_y, delta
params = [1.0, 1.0, 1.0, 1.0, 1.0] # parameters from Demian-Wiggins (2017)
RelTol = 3.e-10
AbsTol = 1.e-10 
end_time = 30
xRes = 100
pxRes = 100
traj_lw = 2
#y_constant = 0
# y_constant = -0.25
y_constant = 0.5

def event_return_sos(t, states, *params):
    if abs(t) < 1e-2:
        val = 10
    else:
        val = states[1] - y_constant
    return val
event_return_sos.terminal = False
event_return_sos.direction = 0


## Obtain the boundary of the energy surface intersection with y = y_constant
xMax_at_yconstant = np.sqrt((total_energy - \
                     (0.5*params[3]**2*y_constant**2 - \
                      (params[4]/3)*y_constant**3))/(0.5*params[2]**2 + y_constant))
xMin_at_yconstant = -xMax_at_yconstant

pxMax_at_yconstant = np.sqrt(2*params[0]*(total_energy - HH2dof.potential_energy(0,y_constant, params[2:])))
pxMin_at_yconstant = -pxMax_at_yconstant


px_at_yconstant = lambda x: np.sqrt(2*params[0]*(total_energy - \
                                    (0.5*params[2]**2*x**2 + 0.5*params[3]**2*y_constant**2 - \
                                     (params[4]/3.0)*y_constant**3 + x**2*y_constant) ))


xGrid_boundary = np.linspace(xMin_at_yconstant + 1e-10, xMax_at_yconstant - 1e-10, 401, endpoint = True)
pxGrid_boundary = px_at_yconstant(xGrid_boundary)
pxGrid_boundary[0] = 0
pxGrid_boundary[-1] = 0

xGrid_boundary = np.append(xGrid_boundary, np.flip(xGrid_boundary))
pxGrid_boundary = np.append(pxGrid_boundary, -np.flip(pxGrid_boundary))

#fig_energy_sos = plt.figure(figsize = (7,7))
#ax_energy_sos = fig_energy_sos.gca()
#plt.plot(xGrid_boundary, pxGrid_boundary, '-r', \
#         linewidth = 2, label = r'$y = $%.3f'%(y_constant))
#ax_energy_sos.set_xlabel(r'$x$', labelpad = 5, rotation = 0, fontsize = 20)
#ax_energy_sos.set_ylabel(r'$p_x$', labelpad = 5, rotation = 0, fontsize = 20)
#ax_energy_sos.legend()

#%% 

xMesh, pxMesh = np.meshgrid(np.linspace(xMin_at_yconstant + 1e-10, xMax_at_yconstant - 1e-10, xRes), \
                           np.linspace(pxMin_at_yconstant + 1e-10, pxMax_at_yconstant - 1e-10, pxRes))

#xMesh, pxMesh = np.meshgrid(xGrid, pxGrid)

xMesh = np.reshape(xMesh, (xRes*pxRes,1))
pxMesh = np.reshape(pxMesh, (xRes*pxRes,1))
yMesh = y_constant*np.ones((xRes*pxRes,1))
pyMesh = np.zeros((xRes*pxRes,1))

pe_cont_vals = [1/6, total_energy]
pe_cont_cols = ['k', 'g']

#fig_traj_pes = plt.figure(figsize = (7,7))
#ax_traj_pes = fig_traj_pes.gca()

#cset = HH2dof.plot_PE_contours(xVec, yVec, params[2:],  \
#                               pe_cont_vals, pe_cont_cols, ax_traj_pes)
#cbar = fig_traj_pes.colorbar(cset, shrink = 0.9, pad = 0.05, \
#                             drawedges = True)
#cbar.ax.tick_params(labelsize = ls_tick - 10)

escape_label = np.empty((xRes*pxRes,1))
escape_label[:] = np.NaN

for i in range(xRes*pxRes):
    pyMesh[i,0] = HH2dof.momentum_fixed_energy(xMesh[i,0], yMesh[i,0], pxMesh[i,0], \
                                               params, total_energy)
    
    if ~np.isnan(pyMesh[i,0]):
        init_cond = [xMesh[i,0], yMesh[i,0], pxMesh[i,0], pyMesh[i,0]]
                        
        sol = solve_ivp(HH2dof.vector_field, [0, end_time], init_cond, \
                        args = params, \
                        events = (HH2dof.event_escape_left, HH2dof.event_escape_right, \
                                  HH2dof.event_escape_top, event_return_sos), \
                        dense_output = True, \
                        rtol = RelTol, atol = AbsTol)
        
        if np.size(sol.t_events) > 0:
#             ax_traj_pes.plot(sol.y[0,:], sol.y[1,:], '-r', linewidth = traj_lw)
#             print(sol.t_events)
#            print(np.size(sol.t_events[3]), sol.t_events[3])
            
            if np.size(sol.t_events[0]) == 1 and np.size(sol.t_events[3]) == 1:
                escape_label[i,0] = 1
                print('Escape via left')
#                ax_traj_pes.plot(sol.y[0,:], sol.y[1,:], '-r', linewidth = traj_lw)
#                print(np.size(sol.t_events[0]),np.size(sol.t_events[3]))
            elif np.size(sol.t_events[1]) == 1 and np.size(sol.t_events[3]) == 1:
                escape_label[i,0] = 2
                print('Escape via right')
#                ax_traj_pes.plot(sol.y[0,:], sol.y[1,:], '-r', linewidth = traj_lw)
#                print(np.size(sol.t_events[1]),np.size(sol.t_events[3]))
            elif np.size(sol.t_events[2]) == 1 and np.size(sol.t_events[3]) == 0:
                escape_label[i,0] = 3
                print('Escape via top')
#                ax_traj_pes.plot(sol.y[0,:], sol.y[1,:], '-r', linewidth = traj_lw)
#                print(np.size(sol.t_events[2]),np.size(sol.t_events[3]))
            else:
                escape_label[i,0] = 0
        else:
            escape_label[i,0] = 0
        
        
# ax_mani_sect.plot(xMesh[~np.isnan(pyMesh)],pxMesh[~np.isnan(pyMesh)], \
#                   '.', markersize = 1)

# plt.savefig('E%.3f'%(total_energy) + '.png', dpi = 300, bbox_inches = 'tight')

# plt.savefig('escape-trajectories_E%.3f'%(total_energy) + '_T%.3f'%(end_time) + '.png', 
#            dpi = 300, bbox_inches = 'tight')

data = np.zeros((xRes*pxRes,5))
data[:,0] = np.squeeze(xMesh)
data[:,1] = np.squeeze(yMesh)
data[:,2] = np.squeeze(pxMesh)
data[:,3] = np.squeeze(pyMesh)
data[:,4] = np.squeeze(escape_label)
np.savetxt('hh_escape_samples%d'%(xRes*pxRes) + \
           '_E%.3f'%(total_energy) + '_T%.3f'%(end_time) + '.txt', \
           data, fmt='%.18e')


#%%


















