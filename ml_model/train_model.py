from resnet_cbam import resnet18, resnet34
from load_data import DataLoader
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix
from f1_metric import F1Score

class TrainModel:
    def __init__(self, 
                 batch=32, 
                 model_type='resnet18', 
                 epoch = 40, 
                 poles=5, 
                 upperCutoff = 15,
                 att_heads=4, 
                 stop_percistance = 30, 
                 fft = False,
                 train_custom_cols = False,
                 custom_cols = ['NORM','MI']):
        
        self.fft = fft
        self.stop_percistance = stop_percistance
        self.att_heads = att_heads
        self.classes = custom_cols if train_custom_cols else ['NORM', 'MI', 'STTC', 'HYP', 'CD']
        self.epoch = epoch
        self.batch = batch
        self.base_path = f'./saves/{model_type}/'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_checkpoint_file = f'{self.base_path}checkpoint/{model_type}_{self.timestamp}.keras'
        os.makedirs(os.path.dirname(self.model_checkpoint_file), exist_ok=True)
        self.model_train_hist = f'{self.base_path}train_history/{model_type}_{self.timestamp}.pkl'
        self.model_type = None
        match model_type:
            case 'resnet18':
                self.model_type = resnet18
            case 'resnet34':
                self.model_type = resnet34
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")
        self.data = DataLoader(poles= poles, upperCutoff = upperCutoff, fft = self.fft, custom_cols= custom_cols)
        self.model = None
        self.__load_model()
        self.X_train, self.X_test, self.X_val = self.data.get_cst_in() if train_custom_cols else self.data.get_flt_in()
        self.y_train, self.y_test, self.y_val = self.data.get_cst_out() if train_custom_cols else self.data.get_class_out()
        self.model_checkpoint = None
        self.history = None

    def __load_model(self):
        heads = 24 if self.fft else 12
        inputs = keras.Input(batch_size=self.batch, shape=( heads, 1000, 1))
        outputs = self.model_type(inputs, num_classes=5, num_heads = self.att_heads)
        self.model = keras.Model(inputs, outputs)

    def train_model(self):
        callbacks_list = [
                keras.callbacks.EarlyStopping(
                    monitor='val_f1_macro',
                    patience= self.stop_percistance,
                    mode='max',
                    restore_best_weights=True,
                    verbose=1),
                keras.callbacks.ModelCheckpoint(filepath=self.model_checkpoint_file, monitor='val_f1_macro', save_best_only=True)
            ]
        opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
        self.model.compile(
            optimizer=opt, 
            loss='binary_crossentropy', 
            metrics=[
                # tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.BinaryAccuracy(name='b_acc'),
                F1Score(name='f1_macro')
            ])
        self.history = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch, callbacks=callbacks_list, validation_data=(self.X_val, self.y_val))
        self.model_checkpoint = keras.models.load_model(self.model_checkpoint_file)
        # self.save_history()

    def test_model(self):
        results = self.model.predict(self.X_test)
        predictions = tf.argmax(results, axis=1).numpy()
        true_labels = tf.argmax(self.y_test, axis=1).numpy()

        # Calculate precision, recall, f1-score
        report = classification_report(true_labels, predictions, target_names=self.classes)
        print("Classification Report:")
        print(report)

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        print("Confusion Matrix:")
        print(" " * 10 + "Predicted")
        print(" " * 8 + " ".join(f"{cls:^8}" for cls in self.classes))
        print("Actual")
        for i, row in enumerate(cm):
            print(f"{self.classes[i]:<8} " + " ".join(f"{val:^8}" for val in row))

        # Extract true positives, true negatives, false positives, false negatives
        tp = cm.diagonal()
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (fp + fn + tp)

        print("\nMetrics per class:")
        for i, cls in enumerate(self.classes):
            print(f"Class: {cls}")
            print(f"  True Positives: {tp[i]}")
            print(f"  True Negatives: {tn[i]}")
            print(f"  False Positives: {fp[i]}")
            print(f"  False Negatives: {fn[i]}")

    def save_history(self):
        with open(self.model_train_hist , 'wb') as f:
            pickle.dump(self.history, f)

    def load_history(self, file):
        with open(file, 'rb') as f:
            self.history = pickle.load(f)
   