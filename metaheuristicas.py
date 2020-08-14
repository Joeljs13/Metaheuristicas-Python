#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:20:49 2020

@author: joel
"""
import numpy as np 
from numpy.random import random_sample
import pandas as pd


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
       self.g_pos = None
           
    def optimize(self, n_part, n_iter):
        """
        This function optimize the function given
        
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
        
        self.g_pos = None
            
        self._createPopulation(n_part)   
      
        self._movePopulation()
        
        
        
    def _createPopulation(self, n_part):
        """
        Creates population of n particles

        Parameters
        ----------
        n_part : int
            Number of particles to create the population with.

        Returns
        -------
        None

        """
        
        columns = ['Position',
                   'Velocity',
                   'Fitness',
                   'Best_pos']
        
        self.population = pd.DataFrame(index=range(n_part),
                                       columns = columns)
        
        dim = len(self.bounds.values())
        
        rand_mat = random_sample((n_part, dim))
        for j, lims in enumerate(self.bounds.values()):
            rand_mat[:,j] = (lims[1] - lims [0])*rand_mat[:,j] + lims[0]

        self.population.loc[:,'Position'] = list(rand_mat)
        self.population.loc[:, 'Best_pos'] = list(rand_mat)
        self.population.loc[:,'Velocity'] = list(random_sample((n_part, dim)))
               
        
        idx_min = self._evaluatePopulation()
        self.g_pos = self.population.iloc[idx_min,0]
        
    def _evaluatePopulation(self):
        """
        Evaluate the current population

        Returns
        -------
        Idx min of the fitness

        """
        self.population.loc[:,'Fitness'] = self.func(self.population.loc[:,'Position'])
        
        return self.population.loc[:,'Fitness'].idxmin()
            
        
    def _movePopulation(self):
        for particle in enumerate(self.population):
            break
                                     
    
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
        pass
   
       
        
  
bounds = {'x': [1,2],
          'y': [9,10],
          'z': [-3,-1]}

def sum(x):
    results = []
    r = 0
    for pos in x:
        r = 0 
        for c in pos:
            r += c
        
        results.append(r)
 
    return results

pso = PSOOptimizer(sum, bounds)
pso.optimize(4,2)
#pso._getPopInfo()
