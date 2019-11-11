# -*- coding: utf-8 -*-
"""
This script provides the code for the bettor class containing 
the betting logic for bettor agents

[to do: follow the scikit-learn commenting format]
"""
import random
import pandas as pd

# the bettor class represents a bettor with a strategy
class bettor():
    def __init__(self, weighted_choice = True, top_n = 1):
        '''status parameters'''
        self.tournament = None      # pointer to the tournament
        self.alive = True           # death flag
        self.picked = []            # list of picks from prevous weeks
        self.weeks_survived = 0     # track the duration of survival

        '''strategy parameters'''
        # top_n is the number of picks the bettor considers choosing 
        # in each week (e.g., top_n = 1 means pick the top pick each week) 
        self.top_n = top_n   
        
        # weighted_choice is a boolean
        # true = select randomly by weight
        # false = select randomly among the top_n assuming equal weighting
        # [to do: consider if the user should pass equal weights
        # instead of keeping this parameter]      
        self.weighted_choice = weighted_choice 
        
    def make_pick(self, season, week, data): 
        '''
        make_pick gets called from the tournament class
        passes along data to the run_model function
        which returns decision logic for making the pick,
        then makes the pick, tracks the pick, and returns the pick.
        make_pick is intented to the relatively static in subclasses
        of the bettor base class

        season is the year
        week is the week
        data is the data in the form of a dataframe
            passed through from the tournament class
            for the base class, this consists of the following columns:
            'schedule_season', 'schedule_week', 
               'team_home_id', 'team_away_id','team_favorite_id', 
               'team_underdog_id', 'spread_favorite'  
        '''
        # run model which returns a dataframe with 2 columns
        # team and bet_strength.
        bet_strengths = self.run_model(season, week, data)
        # remove picked teams
        bet_strengths = bet_strengths[~bet_strengths.team.isin(self.picked)]
        # keep top_n
        bet_strengths = bet_strengths.nlargest(n = self.top_n, columns = 'bet_strength', keep = 'all')
        
        #select
        if self.weighted_choice: 
            pick = random.choices(population=bet_strengths.team.tolist(),
                                  weights=bet_strengths.bet_strength.tolist())[0]
        else:    
            pick = random.choice(bet_strengths.team.tolist())
        
        #track picked
        self.picked.append(pick)
        
        return pick
        

    
    def run_model(self, season, week, data):
        '''
        run_model gets called from the make_pick function
        and is meant to be extended in subclasses with more 
        complicated decision logic.
        run_model recieves data from 
        the tournament class that is passed through the make_pick 
        function.  
        However, I can foresee scenarios where the run_model 
        function would gather additional data or models from the tournament class
        or other sources, or load models from other sources. 
        '''
        
        # in the base class, the dataframe data includes the 
        # columns team_favorite_id and spread_favorite
        # run_model returns a dataframe with the 
        # columns team and bet_strength
        # with bet_strength equal to the absolute value of the spread
        model_results = pd.DataFrame()
        model_results['team'] = data.team_favorite_id
        model_results['bet_strength'] = -data.spread_favorite
        return model_results
        
