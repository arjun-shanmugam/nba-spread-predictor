import tensorflow as tf
from load_and_preprocess import preprocess_features

# model = tf.keras.models.load_model("nba")

# _, _, _, test_ds = preprocess_features(batch_size=256)
# for element in test_ds.as_numpy_iterator():
#     pred = model(element)
#     print(pred)
#     print(element)

teamnames_and_record = [("Hawks", .354),
            ("Celtics", .598),
            ("Cavaliers", .232),
            ("Pelicans", .402),
            ("Bulls", .268),
            ("Mavericks", .402),
            ("Nuggets", .659),
            ("Warriors", .695),
            ("Rockets", .646),
            ("Clippers", .585),
            ("Lakers", .451),
            ("Heat", .476),
            ("Bucks", .732),
            ("Timberwolves", .439),
            ("Nets", .512),
            ("Knicks", .207,),
            ("Magic", .512),
            ("Pacers", .585),
            ("76ers", .622),
            ("Suns", .232),
            ("Trail Blazers", .646),
            ("Kings", .476),
            ("Spurs", .585),
            ("Thunder", .598),
            ("Raptors", .707),
            ("Jazz", .610),
            ("Grizzlies", .402),
            ("Wizards", .390),
            ("Pistons", .500),
            ("Hornets", .476)]

ids = range(1610612737, 1610612767)

id_to_teamname_and_record = dict(zip(ids, teamnames_and_record))
