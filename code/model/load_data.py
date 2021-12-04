import pyreadstat
import numpy as np


def load_data_as_numpy():
    dtafile = '../../cleaned_data/games_and_players.dta'
    df, meta = pyreadstat.read_dta(dtafile)
    df.replace("", float('nan'), inplace=True) #replace empty string with NaN
    df.dropna(axis=0, inplace=True) #remove any row w NaN
    # print(df.head())
    df.to_csv("../../csv_data.csv") #so that I can view the data
    return df.to_numpy()

def split_val_and_train(data):
    np.random.shuffle(data)
    train, validation = data[len(data)//5 * 4], data[len(data)//5*4]
    return train, validation
