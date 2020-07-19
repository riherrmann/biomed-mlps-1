from keras.models import Sequential
from keras.layers import Dense, Dropout
from biomed.properties_manager import PropertiesManager
from keras.regularizers import l1_l2
from biomed.mlp.model_base import ModelBase

class MultiSimpleFFN( ModelBase ):
    def __init__( self, Properties: PropertiesManager ):
        super( MultiSimpleFFN, self ).__init__( Properties )

    def buildModel(self, input_dim, nb_classes) -> str:
        Model = Sequential()
        Model.add(Dense(64, input_dim=input_dim, activation='relu', activity_regularizer=l1_l2(0.001, 0.01)))
        Model.add(Dropout(0.25))
        Model.add(Dense(32, activation='relu', kernel_initializer='random_uniform', bias_initializer='zero'))
        Model.add(Dropout(0.1))
        Model.add(Dense(nb_classes, activation='softmax'))

        Model.compile(
            loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy']
        )

        self._Model = Model
        return self._summarize()
