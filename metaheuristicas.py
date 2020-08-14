#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:20:49 2020

@author: joel
"""
import numpy as np 
from numpy.random import random_sample

class Particle():
    """
    Particle class.
    
    Parameters
    ----------
    pos : Numpy array of size N where N is the dimension of the search
          space.
           Initial position
    vel : Numpy array of size N where N is the dimension of the search
          space.
           Initial velocity
    Returns
    -------
    None.
    """
    def __init__(self, pos, vel):
        
        self.pos = np.copy(pos)
        self.vel = np.copy(vel)
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
       self.population = None
       
    
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
        if self.population:
            self._clearPopulation()
        
        self.population = self._createPopulation(n_part)
        
        
    def _createPopulation(self, n_part):
        """
        Creates population of n particles

        Parameters
        ----------
        n_part : int
            Number of particles to create the population with.

        Returns
        -------
        List with n particles

        """
        population = []
        
        for i in range(n_part):
            pos = np.empty(len(self.bounds.keys()))
            
            for i, lims in enumerate(self.bounds.values()):
                pos[i] = (lims[1] - lims [0])*random_sample() + lims[0]
            
            vel = random_sample(pos.shape)
            
            population.append(Particle(pos,vel)) 
            
        return population

                         
    
    def _clearPopulation(self):
        """
        Makes population == None
        
        Returns
        -------
        None.

        """
        self.population = None 
        
    def _getPopInfo(self):
        """
        Function to get Population information. In the future will be used
        to create optimization logs.

        Returns
        -------
        None.

        """
        for particle in self.population:
            print(particle.pos)
            print(particle.vel)
            
        
  
bounds = {'x': [-2,2],
          'y': [9,10]}

pso = PSOOptimizer(2,bounds)
pso.optimize(4,2)
pso._getPopInfo()
