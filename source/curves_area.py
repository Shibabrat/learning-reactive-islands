
import numpy as np


def greens_poly_area(curve2d):
    """
    Calculates the area of a 2D polygon using the Greens Theorem 
    
    Ref: Eqn 6.14 https://openstax.org/books/calculus-volume-3/pages/6-4-greens-theorem
    """

    closed_curve2d = np.append(curve2d, [curve2d[0,:]], axis = 0) # closing the curve
    diff_curve = np.diff(closed_curve2d, axis = 0) # [dx, dy]
    
    curve2d_shift = np.roll(curve2d, -1, axis = 0)
    
    area = 0
    for i in range(len(curve2d_shift)):
        area = area + 0.5*( -curve2d_shift[i,1]*diff_curve[i,0] + \
                          curve2d_shift[i,0]*diff_curve[i,1] )
        
    return np.fabs(area)




