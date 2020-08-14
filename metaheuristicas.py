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
       self.positions = None
       self.fitness = None
       self.g_pos = None
       
    
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
        
        if self.fitness:
            self.fitness = None
            
        if self.g_pos:
            self.g_pos = None
            
        if self.positions:
            self.positions = None
        
        self.population, self.positions = self._createPopulation(n_part)
        self.fitness = self._evaluatePopulation()
        
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
        List with n particles. Matrix of n x m with n particles in a m 
        dimension space

        """
        population = []
        pop_pos = []
        
        for i in range(n_part):
            pos = np.empty(len(self.bounds.keys()))
            
            for i, lims in enumerate(self.bounds.values()):
                pos[i] = (lims[1] - lims [0])*random_sample() + lims[0]
            
            vel = random_sample(pos.shape)
            
            population.append(Particle(pos,vel)) 
            pop_pos.append(pos)
            
        return (population, np.asarray(pop_pos))

    def _evaluatePopulation(self):
        """
        Evaluate the current population

        Returns
        -------
        fitness : ndarray (n,) where n is the number of particles
            Fitness of every particle

        """
        fitness = np.empty(self.positions.shape[0])
        
        for i, position in enumerate(self.positions):
            fitness[i] = self.func(position)
            
        return fitness 
        
    def _movePopulation(self):
        pass                         
    
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
        print(self.positions.shape)
        print(self.positions)
        print(self.fitness.shape)
        print(self.fitness)
       
        
  
bounds = {'x': [-2,2],
          'y': [9,10],
          'z': [-3,-1]}

def sum(x):
    sum = 0 
    for i in x:
        sum += i
        
    return sum 

pso = PSOOptimizer(sum, bounds)
pso.optimize(4,2)
pso._getPopInfo()

