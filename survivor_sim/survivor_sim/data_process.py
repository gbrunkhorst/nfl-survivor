# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 21:38:53 2019

@author: greg
"""
import pandas as pd
import random
from sklearn.linear_model import LogisticRegression

def load_process_spread_scores(export_csv = False):
   
 
    # load CSV file (data originally from Kaggle
    # https://www.kaggle.com/tobycrabtree/nfl-scores-and-betting-data

    source_path = r'C:\Users\greg\Documents\NFL\survivor_sim\source_data'
    processed_path = r'C:\Users\greg\Documents\NFL\survivor_sim\processed_data'
    # todo: change this to a relative path 

    teams = pd.read_csv(source_path + "\\nfl_teams.csv")
    spread_scores = pd.read_csv(source_path + '\\spreadspoke_scores.csv')
    
    # add columns that use the team IDs rather than the team name
    spread_scores = spread_scores[
            (spread_scores.schedule_season >= 1979) & 
            (spread_scores.schedule_season <= 2018)]
    spread_scores = spread_scores.merge(
            teams[['team_name', 'team_id']], 
            left_on = 'team_home', 
            right_on = 'team_name'
            ).rename(columns = {'team_id': 'team_home_id'})
    spread_scores = spread_scores.merge(
            teams[['team_name', 'team_id']], 
            left_on = 'team_away', 
            right_on = 'team_name'
            ).rename(columns = {'team_id': 'team_away_id'})
    spread_scores.drop(columns = ['team_name_x', 'team_name_y'], inplace = True)
    
    # loop through the rows and add convenience columns
    for idx in spread_scores.index:
        # add winner
        if spread_scores.loc[idx,'score_home'] > spread_scores.loc[idx,'score_away']:
            spread_scores.loc[idx,'winner'] = spread_scores.loc[idx,'team_home_id']
        elif spread_scores.loc[idx,'score_home'] < spread_scores.loc[idx,'score_away']:
            spread_scores.loc[idx,'winner'] = spread_scores.loc[idx,'team_away_id']  
        else:
            spread_scores.loc[idx,'winner'] = 'tie'
        
        # ID if the favorite won
        spread_scores.loc[idx, 'favorite_won'] = (
                    spread_scores.loc[idx,'winner'] == 
                    spread_scores.loc[idx,'team_favorite_id'])
        # if PICK then just put a random boolean for stats
        if spread_scores.loc[idx,'team_favorite_id'] == 'PICK':
            spread_scores.loc[idx, 'favorite_won'] = (
                    random.choice([True, False]))
        
        # add underdog ID
        if (spread_scores.loc[idx,'team_favorite_id'] == 
                    spread_scores.loc[idx,'team_home_id']):
            spread_scores.loc[idx, 'team_underdog_id'] = (
                    spread_scores.loc[idx,'team_away_id'])
        else:
            spread_scores.loc[idx, 'team_underdog_id'] = (
                    spread_scores.loc[idx,'team_home_id'])
    
    if export_csv == True:
        spread_scores.to_csv(processed_path + '\\spreadspoke_scores_processed.csv')
        
    
    return spread_scores

  
