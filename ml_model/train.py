from train_model import TrainModel
import pandas as pd
import tensorflow as tf

model = TrainModel(epoch = 150, filter=True, poles=5, upperCutoff = 15)
model.train_model()
model.test_model()