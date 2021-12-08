from arjunmodel import *



model, train_ds, test_ds = create_model()
model_name = train_and_testmodel(model, train_ds, test_ds)
predict('nba')


