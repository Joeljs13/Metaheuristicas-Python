#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:20:49 2020

@author: joel
"""
import numpy as np 
from numpy.random import random_sample
import pandas as pd

        
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
    k     : Boolean, optional
            Factor with which velocities are multiplied. The default is True
    Returns
    -------
    None.

    """
    
    def __init__(self, func, bounds, w=0.9, s_fac=2.05, c_fac=2.05, k=True):
       self.func = func
       self.bounds = bounds
       self.w = w 
       self.s_fac = s_fac
       self.c_fac = c_fac
       self.population = None
       self.g_pos = None
       self.dec_w = 0
       if k and (s_fac + c_fac > 4):
           phi = s_fac + c_fac 
           self.k = 2/np.abs(2-phi-np.sqrt(phi**2 - 4*phi))
       elif k and (s_fac + c_fac < 4):
           print('If k == True then social factor + cognitive factor must be > 4')
           print('k has been set to False')
           self.k = 0
       else:
           self.k = 0
   
           
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
        self.dec_w = self.w/n_iter
            
        self._createPopulation(n_part)   
        
        self._movePopulation(n_part, self.k)
      
        
        
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
            
        
    def _movePopulation(self, npart, k):
        """
        Moves every particle

        Returns
        -------
        None.

        """

        self._calc_new_velocities(npart, k)        
        self._calc_new_positions(npart)
    
        
    def _calc_new_velocities(self, npart, k):
        """
        Function to calculate new velocities

        Parameters
        ----------
        npart : Int
            Number of particles.
        k : Float
            Factor with which velocites are multiplied.

        Returns
        -------
        None.

        """
        rn = random_sample(2*npart)
        
        if k: 
            for i in range(npart):
                s_f = self.c_fac*rn[2*i]*(self.population.iloc[i,3]-self.population.iloc[i,0])
                p_f = self.s_fac*rn[2*i+1]*(self.g_pos-self.population.iloc[i,0])
                n_vel = k * (self.w*self.population.loc[i,'Velocity'] + s_f + p_f)
                self.population.loc[i,'Velocity'] = [[x] for x in n_vel]
            
        else:
            for i in range(npart):
                s_f = self.c_fac*rn[2*i]*(self.population.iloc[i,3]-self.population.iloc[i,0])
                p_f = self.s_fac*rn[2*i+1]*(self.g_pos-self.population.iloc[i,0])
                n_vel = self.w*self.population.loc[i,'Velocity'] + s_f + p_f
                self.population.loc[i,'Velocity'] = [[x] for x in n_vel]
          
        self.w -= self.dec_w
        
    def _calc_new_positions(self,npart):
        """
        Function to calculate new positions

        Parameters
        ----------
        npart : Int
            Number of particles.

        Returns
        -------
        None.

        """
        for i in range(npart):
            self.population.loc[i,'Position'] = self._checkPosition(i)
            
    def _checkPosition(self,i):
        """
        Check positions and handles bounds violations

        Parameters
        ----------
        i : Int
            Index of particle to check position.

        Returns
        -------
        list
            Position in a way that avoids bugs with Pandas.

        """
        n_pos = self.population.loc[i,'Position'] + self.population.loc[i,'Velocity']
        
        return [[x] for x in n_pos]
        
        
                
    def _clearPopulation(self):
        """
        Makes population == None
        
        Returns
        -------
        None.
        """
        
        self.population = None 
        

    
        
    def _getPopInfo(self, col='Fitness'):
        """
        Function to get Population information. In the future will be used
        to create optimization logs.

        Returns
        -------
        None.

        """
        if col == 'all':
            print(self.population)
        else:
            try:
                print(self.population.loc[:,col])
            except:
                print("'{}' is not a valid option".format(col))
   
       
        
  
bounds = {'x': [1,2],
          'y': [9,10],}
          #'z': [-3,-1]}

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
pso.optimize(2,2)
#pso._getPopInfo('Position')
#pso._getPopInfo('Velocity')
