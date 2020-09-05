#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:20:49 2020

@author: joel
"""
import numpy as np 
from numpy.random import random_sample
from operator import attrgetter


class Particle():
    def __init__(self, pos, vel, b_pos):
        self.pos = pos.copy() 
        self.vel = vel.copy()
        self.b_pos = b_pos.copy()
        self.fitness = np.inf
        self.bp_fitness = np.inf
        
        
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
       self.fg_pos = np.inf
       self.dec_w = 0
       self.history = {}
       if k and (s_fac + c_fac > 4):
           phi = s_fac + c_fac 
           self.k = 2/np.abs(2-phi-np.sqrt(phi**2 - 4*phi))
       elif k and (s_fac + c_fac < 4):
           print('If k == True then social factor + cognitive factor must be > 4')
           print('k has been set to False')
           self.k = 0
       else:
           self.k = 0
   
           
    def optimize(self, n_part, n_iter,b_handling='None'):
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
        self.history = {}
        self.g_pos = None
        self.fg_pos = np.inf
        self.dec_w = self.w/n_iter
        self.i = 0
        
        
        self._createPopulation(n_part) 
        for i in range(n_iter):
            self._movePopulation(n_part, self.k, b_handling)
           # self._getPopInfo('all')
           # print(self.g_pos)
           # print(self.fg_pos)
        
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
        
        
        self.population = []
        
        dim = len(self.bounds.values())
        
        rand_mat = random_sample((n_part, dim))
        for j, lims in enumerate(self.bounds.values()):
            rand_mat[:,j] = (lims[1] - lims [0])*rand_mat[:,j] + lims[0]
        
        for i in range(n_part):
            self.population.append(Particle(rand_mat[i,:], random_sample(dim), rand_mat[i,:]))
   
     
        idx_min = self._evaluatePopulation(n_part)
        if self.population[idx_min].fitness < self.fg_pos:
            self.g_pos = self.population[idx_min].pos
            self.fg_pos = self.population[idx_min].fitness
        self._getPopInfo()
        self.i += 1
    def _evaluatePopulation(self, npart):
        """
        Evaluate the current population

        Returns
        -------
        Idx min of the fitness

        """
        for i in range(npart):
            fitness = self.func(self.population[i].pos)
            self.population[i].fitness = fitness 
            if self.population[i].fitness < self.population[i].bp_fitness:
                self.population[i].b_pos = self.population[i].pos.copy()
                self.population[i].bp_fitness = fitness
        
        return self.population.index(min(self.population, key=attrgetter('fitness')))
            
        
    def _movePopulation(self, npart, k, b_handling):
        """
        Moves every particle

        Returns
        -------
        None.

        """
        
        self._calc_new_velocities(npart, k)        
        self._calc_new_positions(npart, b_handling)
        
        idx_min = self._evaluatePopulation(npart)
        if self.population[idx_min].fitness < self.fg_pos:
            self.g_pos = self.population[idx_min].pos
            self.fg_pos = self.population[idx_min].fitness
        self._getPopInfo()
        self.i += 1
    
        
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
                p_f = self.c_fac*rn[2*i]*(self.population[i].b_pos-self.population[i].pos)
                s_f = self.s_fac*rn[2*i+1]*(self.g_pos-self.population[i].pos)
                n_vel = k * (self.w*self.population[i].vel + s_f + p_f)
                self.population[i].vel = n_vel               
        else:
            for i in range(npart):
                p_f = self.c_fac*rn[2*i]*(self.population[i].b_pos-self.population[i].pos)
                s_f = self.s_fac*rn[2*i+1]*(self.g_pos-self.population[i].pos)
                n_vel = self.w*self.population[i].vel + s_f + p_f
                self.population[i].vel = n_vel 
          
        self.w -= self.dec_w
        
    def _calc_new_positions(self,npart,b_handling):
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
            self.population[i].pos = self._checkPosition(i,b_handling)
            
    def _checkPosition(self,i,b_handling):
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
        if b_handling == 'None':
            n_pos = self.population[i].pos + self.population[i].vel
        elif b_handling == 'inf':
            n_pos = self.population[i].pos + self.population[i].vel
            for k, values in enumerate(self.bounds.values()):
                if self.population[i].pos[k] < values[0] or self.population[i].pos[k] > values[1]:
                    print('f')               
        
        return n_pos.copy()
        
        
                
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
        pos = []
        vel = []
        b_pos = []
        fitness = []
        bp_fitness = []
        for p in self.population:
            pos.append(p.pos)
            vel.append(p.vel)
            b_pos.append(p.b_pos)
            fitness.append(p.fitness)
            bp_fitness.append(p.bp_fitness)
            

            
        self.history[self.i] = {'Pos':pos,'Vel':vel,'B_pos':b_pos,
                                'Fitness':fitness, 'BP_fitness':bp_fitness,
                                'GB_pos':self.g_pos,'GB_fitness':self.fg_pos}
       
        
  
bounds = {'x': [-10,10],
          'y': [-10,10],}
          #'z': [-10,10]}

def sum(x):
    r = 0
    for c in x:
        r += c
    
    return r

def sphere(x):
    r = 0
    for c in x:
        r+= c**2
        
    return r

pso = PSOOptimizer(sum, bounds)
pso.optimize(2,3,b_handling='None')
print(pso.g_pos)




#df = pd.DataFrame.from_dict(pso.history, orient="columns").stack().to_frame()
#df = pd.DataFrame(df[0].values.tolist(), index=df.index)
#df.T
