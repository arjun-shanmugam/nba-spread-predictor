import pandas as pd
import numpy as np
from arjunmodel.load_and_preprocess import df_to_dataset

def preprocess():
    train, test = load_df()
    #extract team data for train
    train_years, train_months, _ = train.pop("year"), train.pop("month"), train.pop("day")
    train_home_team_ids, train_away_team_ids =  train.pop("home_team_id"), train.pop("visitor_team_id")
    #extract team data for test
    test_years, test_months, _ = test.pop("year"), test.pop("month"), test.pop("day")
    test_home_team_ids, test_away_team_ids =  test.pop("home_team_id"), test.pop("visitor_team_id")
    #create the unique ids for train
    train_ids, team_map = create_unique_team_year_id(train_years, train_months,
             train_home_team_ids, train_away_team_ids)
    #create unique ids for test (ids are deterministic, teams will have same id)
    test_ids, team_map = create_unique_team_year_id(test_years, test_months,
             train_home_team_ids, test_away_team_ids, team_map=team_map) #pass in existing map
    train_ds, test_ds = df_to_dataset(train), df_to_dataset(test)
    
    return train_ds, test_ds, train_ids, test_ids, team_map

def create_unique_team_year_id(years, months, team_home_ids, team_away_ids, team_map = None):
    ids = np.zeros((len(team_home_ids), 2))
    if team_map == None:
        team_map = dict()
    count = 0
    idx = 0
    for month, year, home_id, away_id in zip(months, years, team_home_ids, team_away_ids):
        the_year = year #may change year, so create new variable
        if month <= 7 or (year == 2020 and month <= 10): #special case for 2020
            the_year = year - 1 #count the year as the year the season began in
        if (year, home_id) in team_map:
            ids[idx, 0] = team_map[(year, home_id)]
        else:
            team_map[(year, home_id)] = count
            ids[idx, 0] = team_map[(year, home_id)]
            count += 1
        if (year, away_id) in team_map:
            ids[idx, 1] = team_map[(year, away_id)]
        else:
            team_map[(year, away_id)] = count
            ids[idx, 1] = team_map[(year, away_id)]
            count += 1
        idx += 1
    return ids, team_map #create unique id

"""
Loads a stata file as a pd.DataFrame, designates a labels column, and splits into train and test.
"""
def load_df():
    path_to_data = "../../cleaned_data/games_and_players.dta"
    df = pd.read_stata(path_to_data)
    df = df.rename(columns={'point_differential': 'labels'})  # rename point_differential to target (labels)
    train = df[(7 < df['month']) | (df['month'] <= 3)]  # train on games from 2017 and earlier (18779/23597 games)
    test = df[(7 >= df['month']) & (df['month'] > 3)]  # test on games from after 2017 (4818/23597 games)
    return df, test #returning entire dataframe!