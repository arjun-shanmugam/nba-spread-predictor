import tensorflow as tf
from arjunmodel.load_and_preprocess import preprocess_features

model = tf.keras.models.load_model("nba")

_, _, _, test_ds = preprocess_features(batch_size=256)
for element in test_ds.as_numpy_iterator():
    pred = model(element)
    print(pred)
    print(element)