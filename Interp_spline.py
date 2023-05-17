# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:08:15 2022

@author: padil
"""
import numpy as np

def Interp_spline(coords, CoCoeff):

    out = np.zeros((len(coords), 1))
    vec_interp = out

    for idx in range(len(coords)):

        x_tilda = coords[idx,0]
        y_tilda = coords[idx,1]

        x_tilda_floor = np.floor(coords[idx,0])
        y_tilda_floor = np.floor(coords[idx,1] )



        y_d = y_tilda - y_tilda_floor
        x_d = x_tilda - x_tilda_floor

        x_vec = np.array([1.0 , x_d, x_d**2, x_d**3, x_d**4, x_d**5])

        y_vec = np.array([1.0, y_d, y_d**2, y_d**3, y_d**4, y_d**5])

        #print(idx)
                    
        vec_interp[idx,0] = np.matmul(   np.matmul( y_vec,CoCoeff[int(y_tilda)][int(x_tilda)] ) , x_vec  )
                                                #y                  #x
        #vec_interp[idx,0] = np.matmul(   np.matmul( y_vec,CoCoeff[int(y_tilda_floor)][int(x_tilda_floor)] ) , x_vec  )
                                  #CoCoeff = cell array


    return vec_interp
