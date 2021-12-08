import pandas as pd
import tensorflow as tf

"""
***********************************************************************
This file loads data from a .dta file using pandas, converts it to a  *
tf.data.Dataset object, and splits it into testing and training data. *
***********************************************************************
"""


"""
Uses the helper methods below to actually preprocess the features
"""
def preprocess_features(batch_size=256, preprocess_for_embeddings=False):
    # convert train and test DataFrames to Datasets
    train, test = load_df()
    train.drop(['labels'], axis=1)
    test.drop(['labels'], axis=1)
    train_ds = df_to_dataset(train)
    test_ds = df_to_dataset(test)

    # create lists of numerical and categorical feature names
    colnames = train.columns.tolist()  # list of all columns
    print(colnames)
    categorical_features = ['home_team_id', 'visitor_team_id']  # categoricals
    colnames.remove('home_team_id')
    colnames.remove('visitor_team_id')
    numerical_features = colnames  # after removing the only two categorical features, what remains are numerical features

    all_inputs = []
    encoded_features = []

    # normalize numerical features
    index = 0
    for header in numerical_features:
        print(index)
        index += 1
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs.append(numeric_col)
        encoded_features.append(encoded_numeric_col)

    # encode categorical features
    for header in categorical_features:
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='int64')
        encoding_layer = get_category_encoding_layer(name=header,
                                                     dataset=train_ds,
                                                     dtype='int64',
                                                     max_tokens=30)  # for 30 nba teams
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs.append(categorical_col)
        encoded_features.append(encoded_categorical_col)

    all_features = tf.keras.layers.concatenate(encoded_features)
    return all_features, all_inputs, train_ds, test_ds

"""
Loads a stata file as a pd.DataFrame, designates a labels column, and splits into train and test.
"""
def load_df():
    path_to_data = "../../cleaned_data/games_and_players.dta"
    df = pd.read_stata(path_to_data)
    df = df.rename(columns={'point_differential': 'labels'})  # rename point_differential to target (labels)
    train = df[df['year'] <= 2017]  # train on games from 2017 and earlier (18779/23597 games)
    test = df[df['year'] > 2017]  # test on games from after 2017 (4818/23597 games)
    return train, test

"""
Converts a dataframe to a tensorflow Dataset.
"""
def df_to_dataset(dataframe, batch_size=32):
    df = dataframe.copy()
    labels = df.pop('labels')
    df = {key: value[:, tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

"""
Applies feature-wise normalization to numerical features.
"""
def get_normalization_layer(name, dataset):
    # create a Normalization layer for the feature.
    normalizer = tf.keras.layers.Normalization(axis=None)
    # prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])
    # learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer

"""
Encode categorical features as one-hot vectors.
"""
def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    index = tf.keras.layers.IntegerLookup(max_tokens=max_tokens)
    # Prepare a tf.data.Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])
    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)
    # Encode the integer indices.
    encoder = tf.keras.layers.CategoryEncoding(num_tokens=index.vocabulary_size(), output_mode='one_hot')
    # Apply one-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))

