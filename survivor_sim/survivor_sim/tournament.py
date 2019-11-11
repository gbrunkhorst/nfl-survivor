# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:57:34 2019

@author: greg
"""
import pandas as pd
import numpy as np


class tournament():
    # tournament runs a single season
    def __init__(self, bettors, strategies, season, games_outcomes):
        '''
        parameters
        ----------
        bettors: list of instanciated bettors
        strategies: is a list of strings to ID strategies
        season: numerical year
        games_outcomes: is a dataframe with the games and the outcomes
        games_outcomes includes the following columns:
         ['schedule_date', 'schedule_season', 'schedule_week',
         'schedule_playoff', 'team_home', 'score_home', 'score_away',
         'team_away', 'team_favorite_id', 'spread_favorite', 'over_under_line',
         'stadium', 'stadium_neutral', 'weather_temperature', 'weather_wind_mph',
         'weather_humidity', 'weather_detail', 'team_home_id', 'team_away_id',
         'winner', 'favorite_won', 'team_underdog_id']

         [to do: decide if game_columns should have fewer columns and be 
         more general]
        ''' 
        self.bettors = bettors 
        self.season = season
        self.strategies = strategies
        self.games_outcomes = games_outcomes[
                games_outcomes.schedule_season == season]
        
        '''
        tournament tracking
        -------------------
        results: dataframe of bettors that will be populated with the final weeks survived
                    for each bettor
        pick: dataframe of bettors and weeks to track picks 
        '''
        self.results = pd.DataFrame({'bettor': bettors, 
                                     'strategy': strategies})
        self.results.set_index('bettor', inplace=True)  
        self.picks = pd.DataFrame({'bettor': bettors, 
                                     'strategy': strategies})
        self.picks.set_index('bettor', inplace=True)   
        
        self.week = 1  # start the week at 1
        # allow the bettors to point back to the tournament
        # and reset the bettors as needed
        for bettor in bettors:
            bettor.tournament = self
            bettor.alive = True           # death flag
            bettor.picked = []            # list of picks from prevous weeks
            bettor.weeks_survived = 0 
    
    def run(self):
        # weeks are strings in the input dataframe (due to playoffs).  Convert.
        weeks = pd.to_numeric(self.games_outcomes.schedule_week, 
              errors = 'coerce', downcast = 'integer').max()
        # loop through weeks in a season
        for week in range(1,int(weeks)+1):
            # get a dataframe of the current week's games
            games = self.games_outcomes[
                        self.games_outcomes.schedule_week==str(week)
                            ][['schedule_season', 'schedule_week', 
                            'team_home_id', 'team_away_id','team_favorite_id', 
                            'team_underdog_id', 'spread_favorite']] 
            # loop through the bettors
            for bettor in self.bettors:
                if bettor.alive == True:
                    # make and log a pick
                    pick = bettor.make_pick(season = self.season, week = week, 
                                        data = games)
                    self.picks.loc[bettor,week] = pick
                    # if the pick lost, kill the bettor and record outcome
                    if pick not in (self.games_outcomes[
                                self.games_outcomes.schedule_week==str(week)
                                ]['winner'].tolist()):
                        bettor.alive = False
                        bettor.weeks_survived = week-1
                else: # if dead
                    self.picks.loc[bettor,week] = np.nan
        # after looping through all the weeks, populate the results dataframe
        for bettor in self.bettors:
            self.results.loc[bettor,'season'] = self.season
            self.results.loc[bettor, 'weeks_survived'] = bettor.weeks_survived
            self.results['winnings'] = self.results.weeks_survived == self.results.weeks_survived.max()
            self.results.winnings = self.results.winnings / self.results.winnings.sum()
        # return the two dataframe of results
        #  [to do: consider if this returns nothing, just runs the tournament
        #  and the get_ functions return stuff of interest]    
        return self.results, self.picks
            
    def get_results(self):
        # this is stupid since you could just call
        # object_name.results instead
        # maybe it could turn into a usefull function
        return self.results
