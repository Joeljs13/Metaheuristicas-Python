#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:20:49 2020

@author: joel
"""
import numpy as np 

class Particle():
    """
    Particle class.
    
    Parameters
    ----------
    pos : Numpy array or list of size N where N is the dimension of the search
          space.
           Initial position
    vel : Numpy array or list of size N where N is the dimension of the search
          space.
           Initial velocity
    Returns
    -------
    None.
    """
    def __init__(self, pos, vel):
        pass

        
        self.pos = pos 
        self.vel = vel
        self.b_pos = None
        
        
class PSOOptimizer():
    """
    PSO class. 
    
    Parameters
    ----------
    func : Python Function
            Function to be optimized. Must return a scalar (fitness)
    
    bounds : Dict
            Bounds of every dimension. Must be a dict of type 
            { 'name_of_dimension': [inf_lim, sup_lim] }
    
    w : Scalar, optional
            Inertia factor. The default is 0.9.
    
    s_fac : Scalar, optional
            Social factor of the particles. The default is 2.05.
    c_fac : Scalar, optional
            Cognitive factor of the particles. The default is 2.05.

    Returns
    -------
    None.

    """
    
    def __init__(self, func, bounds, w=0.9, s_fac=2.05, c_fac=2.05):
       self.func = func
       self.bounds = bounds
       self.w = w 
       self.s_fac = s_fac
       self.c_fac = c_fac
       
    
    def optimize(self, n_part, n_iter):
        """
        

        Parameters
        ----------
        n_part : int
            Number of particles to use during optimization
        n_iter : int
            Number of generations to use during optimization

        Returns
        -------
        None.

        """
        pass