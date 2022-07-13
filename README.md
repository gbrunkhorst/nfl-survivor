# nfl-survivor
Repo for simulations and stats for evaluating picking strategies in an NFL survivor pool.

My brother, dad, and I have spent too many years baselessly theorizing about the NFL survivor pool; this is the antidote.  

The repo currently consists of two directories:
* bettors_probs: a module for calculating the strength of a pick based on the what other bettors are picking.    
* survivor_sim: a betting simulation framework for back-testing picking strategies.

Each directory also contains source_data, processed_data, and notebooks for analysis.  

# Survivor Pools
A survivor pool is a betting game that lasts the duration of the NFL season.  
* Each bettor chooses one team to win each week.  
* If the team looses, the bettor is knocked out.  
* The bettor cannot choose the same team twice.  
* The last bettor alive wins the pot.  

# Data
The starting dataset for the simulation is from https://www.kaggle.com/tobycrabtree/nfl-scores-and-betting-data

# How-to
The notebooks folders contain jupyter notebook files that show the logic of the modules
