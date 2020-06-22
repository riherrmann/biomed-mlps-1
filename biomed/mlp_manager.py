from keras import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np


class MLPManager:
    def __init__(self, properties_manager):
        self.properties_manager = properties_manager
        self.model = None

    def build_mlp_model_1(self, input_dim, nb_classes):
        props = self.properties_manager.mlp_model_1_properties['building_properties']
        model = Sequential()
        model.add(Dense(props['dense_1'], input_dim=input_dim))
        model.add(Activation(props['activation_1']))
        model.add(Dropout(props['dropout_1']))
        model.add(Dense(props['dense_2']))
        model.add(Activation(props['activation_2']))
        model.add(Dropout(props['dropout_2']))
        model.add(Dense(nb_classes))
        model.add(Activation(props['activation_final']))
        model.compile(loss=props['loss'], optimizer=props['optimizer'])
        # dot_img_file = 'model.png'
        # tensorflow.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        self.model = model

    def train_and_run_mlp_model_1(self, X_train, X_test, Y_train):
        print("Training...")
        train_props = self.properties_manager.mlp_model_1_properties['training_properties']
        self.model.fit(x=X_train, y=Y_train,
                       epochs=train_props['epochs'],
                       batch_size=train_props['batch_size'],
                       validation_split=train_props['validation_split'],
                       workers=self.properties_manager.workers,
                       use_multiprocessing=True if self.properties_manager.workers > 1 else False
                       )

        print("Generating test predictions...")
        pred_props = self.properties_manager.mlp_model_1_properties['prediction_properties']
        # predictions = self.model.predict_classes(X_test, verbose=pred_props['verbose'])
        if len(Y_train[0]) > 2:
            predictions = np.argmax(self.model.predict(X_test,
                                                       workers=self.properties_manager.workers,
                                                       use_multiprocessing=True if self.properties_manager.workers > 1 else False
                                                       ), axis=-1)
        else:
            predictions = self.model.predict_classes(X_test)
        return predictions
