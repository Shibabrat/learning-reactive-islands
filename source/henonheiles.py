import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

import matplotlib as mpl
label_size = 25 #10, 20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
mpl.rcParams['axes.labelsize'] = 35 #, 15


def potential_energy(x, y, params_pe):
    """
    Potential energy function for the 2 DOF model
    x, y: N x N array of position values as meshgrid data
    params: 1 x 4 array of parameters
    """
    if np.size(x) == 1:
        nX = 1
        nY = 1
    else:
        nX = np.size(x,1)
        nY = np.size(x,0)
        x = np.ravel(x, order = 'C')
        y = np.ravel(y, order = 'C')
    
    omega_x, omega_y, delta  = params_pe
    
    vx = 0.5*omega_x**2*x**2
    vy = 0.5*omega_y**2*y**2 - (delta/3.0)*y**3
    vxy = x**2*y
    
    pe = np.reshape(vx + vy + vxy, (nY, nX))
#     print(np.size(vy))
#     print(x,y,vyx)
    
    return pe


def vector_field(t, states, *params):
    
    massA, massB, omega_x, omega_y, delta = params
    x, y, px, py = states
    
    xDot = px/massA
    yDot = py/massB
    pxDot = - ( omega_x**2*x + 2*x*y )
    pyDot = - ( omega_y**2*y + x**2 - delta*y**2 )
    
    return np.array([xDot, yDot, pxDot, pyDot])


def vector_field_ld(t, states, *params):
    
    massA, massB, omega_x, omega_y, delta = params
    x, y, px, py, ld = states
    p = 0.5
    
    xDot = px/massA
    yDot = py/massB
    pxDot = - ( omega_x**2*x + 2*x*y )
    pyDot = - ( omega_y**2*y + x**2 - delta*y**2 )
    ldDot = np.sum(abs(xDot)**p + abs(yDot)**p + abs(pxDot)**p + abs(pyDot)**p)
    
    
    return np.array([xDot, yDot, pxDot, pyDot, ldDot])

# Obtain one of the momenta coordinate based on input values of other coordinates on the fixed energy surface
def momentum_fixed_energy(x, y, px, params, E):

#     print(params)
    massA, massB, omega_x, omega_y, delta = params
    py = 0
    
    potential_energy_val = 0.5*omega_x**2*x**2 \
        + 0.5*omega_y**2*y**2 - (delta/3.0)*y**3 + x**2*y

    if (E >= (potential_energy_val + (1/(2.0*massA))*(px**2.0))):
        py = np.sqrt( 2.0*massB*(E - (potential_energy_val \
            + (1/(2.0*massA))*(px**2.0)) ) )
    else:
        py = np.NaN
        # print("Momentum isn't real!")
    
    return py 


def event_escape_left(t, states, *params):
    return states[0] - (-1.25)
event_escape_left.terminal = True
event_escape_left.direction = 0

def event_escape_right(t, states, *params):
    return states[0] - (1.25)
event_escape_right.terminal = True
event_escape_right.direction = 0

def event_escape_top(t, states, *params):
    return states[1] - (1.25)
event_escape_top.terminal = True
event_escape_top.direction = 0



def plot_PE_contours(xVec, yVec, params, pe_cont_vals, pe_cont_cols, ax_pes):            

    xMesh, yMesh = np.meshgrid(xVec, yVec)

    pe_surf = potential_energy(xMesh, yMesh, params)
    # pe_clines = np.linspace(totalEnergyEqPt1, 2.0, 30, endpoint = True)
#     pe_clines = [np.linspace(-2, -0.2, 10), totalEnergyEqPt1, 0, \
#                  0.25, 0.5, totalEnergyEqPt3, np.linspace(1.25, 2, 10)]
    
#     plt.close('all')
#     fig_pes = plt.figure(figsize=(10,10))
#     ax_pes = fig_pes.gca()
#     cset = ax_pes.contour(xMesh, yMesh, np.log(pe_surf), 
#                            np.linspace(0, 30, 200, endpoint = True), 
#                            linewidths = 1.9, 
#                            cmap = cm.viridis, alpha = 0.9)
    
#     cset = ax_pes.contourf(xMesh, yMesh, pe_surf, \
#                           pe_clines, \
#                           cmap = cm.coolwarm, alpha = 0.9)

    # cset = ax_pes.contourf(xMesh, yMesh, pe_surf, \
    #                     np.linspace(0, 0.20, 30, endpoint = True), \
    #                     cmap = cm.RdBu_r, alpha = 1.0)
    
    ax_pes.contour(xMesh, yMesh, pe_surf, levels = pe_cont_vals, \
                   colors = pe_cont_cols, linewidths = 1.0)
    
    # ax_pes.contour(xMesh, yMesh, pe_surf, levels = [totalEnergyEqPt1, totalEnergyEqPt3], \
    #                 colors='k', alpha = 1.0)
    # ax_pes.scatter(eq_pt_1[0], eq_pt_1[1], s = 50, c = 'g')
    # ax_pes.scatter(eq_pt_2[0], eq_pt_2[1], s = 50, c = 'g')
    # ax_pes.scatter(eq_pt_3[0], eq_pt_3[1], s = 100, c = 'r', marker = 'X')
    

#     cset = ax_pes.plot_surface(xMesh, yMesh, pe_surf, \
#                                rstride=1, cstride=1, \
#                                cmap = cm.coolwarm, \
#                                linewidth=0, antialiased=True, \
#                                alpha = 0.9)

#     ax_pes = fig_pes.add_subplot(111, projection = '3d')
#     cset = ax_pes.plot_surface(xMesh, yMesh, pe_surf)
    
#     ax_pes.scatter(eq_pt_left[0], eq_pt_left[1], s = 40, c = 'r', marker = 'x')
#     ax_pes.scatter(eq_pt_right[0], eq_pt_right[1], s = 40, c = 'r', marker = 'x')
#     ax_pes.scatter(eq_pt_top[0], eq_pt_top[1], s = 40, c = 'r', marker = 'x')

#     ax_pes.set_aspect('equal')
    ax_pes.set_ylabel(r'$y$', labelpad = 5, rotation = 0)
    ax_pes.set_xlabel(r'$x$', labelpad = 0)
#     ax_pes.set_xticks([-1.5, -0.75, 0.0, 0.75, 1.5])
#     ax_pes.set_yticks([-1.5, -0.75, 0.0, 0.75, 1.5])

#     ax_pes.yaxis.set_ticklabels([])
#     ax_pes.zaxis.set_ticklabels([])

#     ax_pes.zaxis.set_ticks(np.arange(0, 12e4, 2e3))
#     ax_pes.ticklabel_format(axis='z', style='sci', scilimits=(0,0))
#     ax_pes.set_zlabel(r'$V_{\rm DB}(x, y)$', labelpad = 15)
#     np.arange(0, 12e4, 2e3)
        
#     cbar.set_label(r'$V_{DB}(x, y)$',fontsize = ls_axes)

    
    return ax_pes


def energy_boundary_sos_xpx(params, total_energy, y_constant, res = 600):
    """
    Returns energy boundary on the surface of section at a given value of y
    and total energy
    """
    
    xMax_at_yconstant = np.sqrt((total_energy - \
                     (0.5*params[3]**2*y_constant**2 - \
                      (params[4]/3)*y_constant**3))/(0.5*params[2]**2 + y_constant))
    xMin_at_yconstant = -xMax_at_yconstant

#     pxMax_at_yconstant = np.sqrt(2*params[0]*(total_energy - HH2dof.potential_energy(0,y_constant, params[2:])))
#     pxMin_at_yconstant = -pxMax_at_yconstant
        
    px_at_yconstant = lambda x: np.sqrt(2*params[0]*(total_energy - \
                                    (0.5*params[2]**2*x**2 + 0.5*params[3]**2*y_constant**2 - \
                                     (params[4]/3.0)*y_constant**3 + x**2*y_constant) ))

    
    xGrid_boundary = np.linspace(xMin_at_yconstant + 1e-10, xMax_at_yconstant - 1e-10, int(res/2) + 1, \
                                 endpoint = True)
    pxGrid_boundary = px_at_yconstant(xGrid_boundary)
    pxGrid_boundary[0] = 0
    pxGrid_boundary[-1] = 0

    xGrid_boundary = np.append(xGrid_boundary, np.flip(xGrid_boundary))
    pxGrid_boundary = np.append(pxGrid_boundary, -np.flip(pxGrid_boundary))
    
    return np.array([xGrid_boundary,pxGrid_boundary]).T

def get_ris_data(datapath_ris, total_energy, manifold_time, direction = 'forward'):
    """
    Load the reactive islands data computed from globalization of stable and unstable 
    manifolds
    """
    
    if direction == 'forward':
        manifold = 'stable'
    elif direction == 'backward':
        manifold = 'unstable'
        
            
    if total_energy > 0.17:
        ri_topsaddle = np.loadtxt(datapath_ris \
                                     + 'xeU1_' + manifold + '_branch1_eqPt2_DelE%.6f'%(total_energy - 1/6) + '_t' \
                                     + str(manifold_time) + '.txt')
        ri_leftsaddle = np.loadtxt(datapath_ris \
                                      + 'xeU1_' + manifold + '_branch1_eqPt3_DelE%.6f'%(total_energy - 1/6) + '_t' \
                                      + str(manifold_time) + '.txt')
        ri_rightsaddle = np.loadtxt(datapath_ris \
                                       + 'xeU1_' + manifold + '_branch-1_eqPt4_DelE%.6f'%(total_energy - 1/6) + '_t' \
                                       + str(manifold_time) + '.txt')
    else:
        ri_topsaddle = np.loadtxt(datapath_ris \
                                  + 'xeU1_' + manifold + '_branch1_eqPt2_DelE%.7f'%(total_energy - 1/6) + '_t' \
                                  + str(manifold_time) + '.txt')
        ri_leftsaddle = np.loadtxt(datapath_ris \
                                   + 'xeU1_' + manifold + '_branch1_eqPt3_DelE%.7f'%(total_energy - 1/6) + '_t' \
                                   + str(manifold_time) + '.txt')
        ri_rightsaddle = np.loadtxt(datapath_ris \
                                     + 'xeU1_' + manifold + '_branch-1_eqPt4_DelE%.7f'%(total_energy - 1/6) + '_t' \
                                     + str(manifold_time) + '.txt')
    

    return ri_topsaddle, ri_leftsaddle, ri_rightsaddle






    
    