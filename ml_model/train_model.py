from resnet_cbam import resnet18, resnet34
from load_data import DataLoader
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import pickle
import os
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from f1_metric import F1Score
import numpy as np

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
                    custom_cols = ['NORM','MI'],
                    custom_cls_trs = 0.2,
                    balence_custom_cls = False,
                    threshold=0.5):

        self.threshold=threshold
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
        self.data = DataLoader(
            poles= poles, 
            upperCutoff = 
            upperCutoff, 
            fft = self.fft, 
            custom_cols= custom_cols,
            custom_cls_trs=custom_cls_trs,
            balence_custom_cls=balence_custom_cls)
        self.model = None
        self.__load_model()
        self.X_train, self.X_test, self.X_val = self.data.get_cst_in() if train_custom_cols else self.data.get_flt_in()
        self.y_train, self.y_test, self.y_val = self.data.get_cst_out() if train_custom_cols else self.data.get_class_out()
        self.model_checkpoint = None
        self.history = None

    def __load_model(self):
        heads = 24 if self.fft else 12
        inputs = keras.Input(batch_size=self.batch, shape=( heads, 1000, 1))
        outputs = self.model_type(inputs, num_classes=len(self.classes), num_heads = self.att_heads)
        self.model = keras.Model(inputs, outputs)

    def train_model(self):
        callbacks_list = [
                keras.callbacks.EarlyStopping(
                    monitor='val_b_acc',
                    patience= self.stop_percistance,
                    mode='max',
                    restore_best_weights=True,
                    verbose=1),
                keras.callbacks.ModelCheckpoint(filepath=self.model_checkpoint_file, monitor='val_b_acc', save_best_only=True)
            ]
        opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
        self.model.compile(
            optimizer=opt, 
            loss='binary_crossentropy', 
            metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.BinaryAccuracy(name='b_acc'),
            F1Score(name='f1_macro')
            ])
        self.history = self.model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch, callbacks=callbacks_list, validation_data=(self.X_val, self.y_val))
        self.model_checkpoint = keras.models.load_model(self.model_checkpoint_file)
        # self.save_history()

    def test_model(self):
        # Predict probabilities
        results = self.model.predict(self.X_test)

        # Apply threshold to get binary predictions
        predictions = (results >= self.threshold).astype(int)
        true_labels = self.y_test.astype(int)  # assuming y_test is already multi-hot

        # Ensure shapes are (n_samples, n_classes)
        if predictions.ndim == 1:
            predictions = predictions[:, np.newaxis]
            true_labels = true_labels[:, np.newaxis]

        # Handle single-class edge case
        if len(self.classes) == 1:
            print("Single class detected. Skipping target_names to avoid mismatch.")
            report = classification_report(true_labels, predictions, zero_division=0)
        else:
            report = classification_report(true_labels, predictions, target_names=self.classes, zero_division=0)

        print("Classification Report:")
        print(report)

        # Multi-label confusion matrix
        cm = multilabel_confusion_matrix(true_labels, predictions)

        print("Confusion Matrices per class:")
        for i, cls in enumerate(self.classes):
            tn, fp, fn, tp = cm[i].ravel()
            print(f"\nClass: {cls}")
            print(f"  True Positives: {tp}")
            print(f"  True Negatives: {tn}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")

    def save_history(self):
        with open(self.model_train_hist , 'wb') as f:
            pickle.dump(self.history, f)

    def load_history(self, file):
        with open(file, 'rb') as f:
            self.history = pickle.load(f)
   