# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 05:48:25 2019

@author: gbrunkhorst

Feature Engineering
"""

import pandas as pd
import numpy as np

#data upload
path = r'..\processed_data'
data = pd.read_csv(path + '\\spreadspoke_scores_processed.csv')

# more data preparation 
def process(data):
    # base the "left hand" team based on the home team; make the spread negative = favorite   
    data.loc[(data.team_home_id == data.team_favorite_id),'spread_home'] = data['spread_favorite']
    data.loc[(data.team_home_id != data.team_favorite_id),'spread_home'] = -data['spread_favorite']
    # add a flag if the home team won
    data['home_won'] = (data.winner == data.team_home_id)
    
    # replace the playoffs with numbers
    playoff_list = data.schedule_week.drop_duplicates().sort_values()[-6::].tolist()
    number_playoffs = {'Conference':21, 'Division':20, 'SuperBowl':22, 
                       'Superbowl':22, 'WildCard':19, 'Wildcard':19}
    data.loc[data.schedule_week.isin(playoff_list), 'schedule_week'] = data.loc[
            data.schedule_week.isin(playoff_list), 'schedule_week'].map(number_playoffs)
    # change fields to numeric
    data.schedule_week = pd.to_numeric(data.schedule_week)
    data.score_home = pd.to_numeric(data.score_home)
    data.score_away = pd.to_numeric(data.score_away)

    # reorganize the DF so each team is listed once (each game is listed twice)
    data = data[['schedule_season', 'schedule_week', 'team_home_id', 
                  'team_away_id', 'spread_home', 
                  'score_home', 'score_away','home_won']].sort_values(
                  by = ['schedule_season', 'schedule_week', 'team_home_id'])
    data['home'] = True
    data.rename(columns = {'team_home_id':'team', 'team_away_id': 'opponent', 
                            'spread_home':'spread',
                            'score_home':'pts_for', 'score_away': 'pts_against',
                            'home_won' : 'won'}, inplace = True)
    copy = data.copy()
    copy.rename(columns = {'team':'opponent', 'opponent':'team', 
                           'pts_for': 'pts_against', 'pts_against':'pts_for'},
                inplace = True)
    copy.home = False
    copy.spread = -copy.spread
    copy.won = -copy.won
    data = pd.concat([data, copy]).sort_values(by = ['schedule_season', 
                    'schedule_week', 'team'])
    data = data[[ 'schedule_season', 'schedule_week','team','opponent', 'home',
                 'spread','pts_for', 'pts_against', 'won']]
    return data

def lookback_fun(df, impute, params, lookbacks):
    ''' helper function for other feature creators'''
    
    new_features = []
    teams = df.team.drop_duplicates().sort_values().tolist()
    # loop through the lookbacks
    for lookback in lookbacks:
        # lookbacks for points for and points againe
        for param in params:
            new_feature = param+'_roll_'+str(lookback)
            new_features.append(new_feature)
            # get the rolling average for each team
            for team in teams:
                rolling  = df[df.team == team][param].rolling(
                        window = lookback, min_periods = 1).mean().tolist()
                # rolling average is inclusive - shift back and impute the first value as the global average
                rolling.insert(0, impute)
                del rolling[-1]
                df.loc[df.team == team, new_feature] = rolling

    # add opponents' pts for and against
    opp_features = []
    for feature in new_features:
        opp_features.append('opp_'+feature)
    col_names = {'team':'opponent'}
    col_names.update(dict(zip(new_features, opp_features)))
    df = df.merge(df[['schedule_season', 'schedule_week', 'team']+new_features
                     ].rename(columns  = col_names), 
                on = ['schedule_season', 'schedule_week', 'opponent'] )
    return df, new_features + opp_features


def for_against(df, lookbacks = [1]):
    '''function adds features based on points scored and 
    given up for a number of previous games'''
    # df is the starting dataframe
    # lookbacks is the # of games to lookback 
    # the length of lookbacks is the # of new features 
    # to add
    
    average_pts_for = df.pts_for.mean() # for imputing the first value
    params = ['pts_for', 'pts_against']
    return lookback_fun(df, impute = average_pts_for, 
                    params = params, lookbacks = lookbacks)

# add win/loss in previous n games
def win_loss(df, lookbacks = [1]):
    '''function adds features based on win/loss
    for a number of previous games
    calls the prediction function and 
    prints the results'''
    # df is the starting dataframe
    # lookbacks is the # of games to lookback 
    # the length of lookbacks is the # of new features 
    # to add 
    
    average_val = 0.5   # for imputing the first value
    params = ['won']
    return lookback_fun(df, impute = average_val, 
                    params = params, lookbacks = lookbacks)

def for_against_weighted(df, lookbacks = [1], lookbacks_initial = [14]):
    '''
    function adds features based on points scored and 
    given up for a number of previous games
    
    df: is the starting dataframe
    lookbacks: is the # of games to lookback the length of new features to add
    lookbacks_initial: is the initial games to lookback to guage the strength of the team
        needs to be a list of one #
    ''' 
    
    average_pts_for = df.pts_for.mean() # for imputing the first value
    params = ['pts_for', 'pts_against']
    
    # initial loop through the lookbacks is for the baseline
    # strength only.  Used 14 as the loopback
    df = lookback_fun(df, impute = average_pts_for, 
                    params = params, lookbacks = lookbacks_initial)

    # add the adjusted pts_for and against
    df['pts_for_adj'] =  df.pts_for - df.opp_pts_against_roll_14
    df['pts_against_adj'] = df.pts_against - df.opp_pts_for_roll_14
    
    # now use the adjusted points for and against add opponents' pts for and against
    
    params = ['pts_for_adj', 'pts_against_adj']
    opp_features = []
    for feature in new_features:
        opp_features.append('opp_'+feature)
    col_names = {'team':'opponent'}
    col_names.update(dict(zip(new_features, opp_features)))
    games = games.merge(games[['schedule_season', 'schedule_week', 'team']+new_features
                     ].rename(columns  = col_names), 
                on = ['schedule_season', 'schedule_week', 'opponent'] )
    
    
    average_pts_for = games.pts_for_adj.mean() # for imputing the first value
    
    params = ['pts_for_adj', 'pts_against_adj']
    new_features = []

    # loop through the lookbacks
    for lookback in lookbacks:
        # lookbacks for points for and points againe
        for param in params:
            new_feature = param+'_roll_'+str(lookback)
            new_features.append(new_feature)
            # get the rolling average for each team
            for team in teams:
                rolling  = games[games.team == team][param].rolling(window = lookback, min_periods = 1).mean().tolist()
                # rolling average is inclusive - shift back and impute the first value as the global average
                rolling.insert(0,average_pts_for)
                del rolling[-1]
                games.loc[games.team == team, new_feature] = rolling

    
    
    # add opponents' pts for and against (repeated from above, not good - use a separate function)
    opp_features = []
    for feature in new_features:
        opp_features.append('opp_'+feature)
    col_names = {'team':'opponent'}
    col_names.update(dict(zip(new_features, opp_features)))
    games = games.merge(games[['schedule_season', 'schedule_week', 'team']+new_features
                     ].rename(columns  = col_names), 
                on = ['schedule_season', 'schedule_week', 'opponent'] )
    
    






data = process(data)
data = for_against(data, lookbacks = [1,2])
data = win_loss(data, lookbacks = [1,2])


data.to_csv('feature_engineer.csv')









