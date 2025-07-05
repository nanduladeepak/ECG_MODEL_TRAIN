from train_model import TrainModel
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# import tensorflow as tf

model_training_r34 = TrainModel(model_type='resnet18',
                                epoch = 500, 
                                stop_percistance= 20, 
                                att_heads=4, 
                                poles=5, 
                                upperCutoff = 45, 
                                train_custom_cols= True, 
                                custom_cls_trs= 0.3,
                                custom_cols = ['MI'])
model_training_r34.train_model()
model_training_r34.test_model()
sns.relplot(data=pd.DataFrame(model_training_r34.history.history), kind='line', height=4, aspect=4)
plt.savefig('./saves/plots/model_training_r34_training_history_plot.png')