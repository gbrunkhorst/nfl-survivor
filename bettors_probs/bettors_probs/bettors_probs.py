# -*- coding: utf-8 -*-
"""
bettors_probs

class to analyze the probabilities in a survivor pool given the 
behavior of the other bettors

@author: greg
"""
import pandas as pd
import numpy as np
from itertools import product
import pickle
from sklearn.linear_model import LogisticRegression
import datetime

class BettorsProbs():
    def __init__(self):
        # BettorsProbs doesn't ever maintain a state.  It is really just some related functions
        # and need not be a class
        pass

    def probs_spreads(self, spreads, model):
        """Calculate probabilities from spreads
       
        Parameters
        ----------
        spreads : pandas series of spreads (negative)
            [action item: test is can accept other objects (lists, np.arrays)]
        model: pre-trained scikit-learn linear regression model
            [action item:  not sure if this would accept other types of 
            scikit-learn models trained on spreads]
       
        Returns
        -------
        probs : array of probabilities of winning
       
        """
        spreads = spreads.values.reshape(-1, 1)
        probs = model.predict_proba(spreads)[:,1]
        return probs



    def bet_strength(self, bettors, probs = None, spreads = None, model = None, tie_all_lose = False):
        """Calculate the bet strength based on the 
        probabilities and the bettors
       
        Parameters
        ----------
        bettor: pandas series of number of bettors.  Both fractions and whole numbers work.
        probs : pandas series of probabilities.  If None, then probs_spreads is run from spreads.
        spreads: pandas series of spreads (negative).  Only used if probs not passed.
        model: pre-trained scikit-learn linear regression model.  Only used if probs not passed.
        
        [action item: test is can accept other objects (lists, np.arrays)]
       
        Returns
        -------
        bet_strength: array of the strength of each bet expressed as a factor of the 
        probability of winning prior to the bet       
        """
        if probs is None:
            probs = self.probs_spreads(spreads = spreads, model = model)
            probs = pd.Series(probs)
        
        top_n = len(bettors)

        #generate all outcome combinations
        combs = np.array(list(product([0,1], repeat = top_n)), dtype=int)
        # probability for each game outcome
        all_probs = probs.values*combs + (
                    1 - probs.values)*(1 - combs)
        #probability of any combination
        comb_probs = np.prod(all_probs, axis = 1)
        # remaining bettors for each combination
        rem_bet = bettors.values*combs
        # new tournament odds for each outcome
        cond_prob = np.where(rem_bet>0, 1, 0)/rem_bet.sum(axis = 1)[:, np.newaxis]
        # assume they tie if they all lose
        cond_prob[0, :] =  (1/bettors.sum())
  
        if tie_all_lose == True:
            # assume they tie if they all lose
            cond_prob[0, :] =  (1/bettors.sum())
        else:
            # or assume that they all lose if all lose
            cond_prob = np.nan_to_num(cond_prob)
        # new tournament odds times the chance of the outcome occurring
        weight_prob = cond_prob * comb_probs[:, np.newaxis]
        prob_before = 1/bettors.sum()
        prob_if_survive = np.sum(weight_prob, axis = 0)
        bet_strength = prob_if_survive/prob_before
        return bet_strength



     # This function finds an equilibrium on a series of probabilities and bettors using brute force
    def find_equilibrium(self, probs = None, bettors = None, tie_if_all_lose = False):
        # make total bettors = 100 for convenience
        bettors = bettors/bettors.sum()*100
        bet_strength = self.bet_strength(bettors  = bettors, probs = probs)

        #[this is likely pretty ugly - there must be a better way to construct the 
        # solver, but it is sufficient for the work we are doing]
        for increment in [1, .1, .01, .001]:
            # loop through and adjust down toward one as we go
            for pick in bettors.index:
                while bet_strength[pick] > 1:
                    bet_strength = self.bet_strength(bettors  = bettors, probs = probs)
                    bettors[pick] += increment
                    bettors[~bettors.index.isin([pick])] += -(increment * bettors[~bettors.index.isin([pick])] / 
                                bettors[~bettors.index.isin([pick])].sum())
            # go the other way
            for pick in bettors.index:
                while bet_strength[pick] < 1:
                    bet_strength = self.bet_strength(bettors  = bettors, probs = probs)
                    bettors[pick] += -increment
                    others_bool = ~bettors.index.isin([pick])
                    bettors[others_bool] += (increment * bettors[others_bool] / 
                                bettors[others_bool].sum())
           
        # one more normalization to 100
        bettors = bettors/bettors.sum()*100
        bet_strength = self.bet_strength(bettors  = bettors, probs = probs)
        return bettors, bet_strength