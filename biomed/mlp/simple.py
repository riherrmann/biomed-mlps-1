from keras.models import Sequential
from keras.layers import Dense, Input
from biomed.properties_manager import PropertiesManager
from biomed.mlp.mlp import MLP
from biomed.mlp.mlp import MLPFactory

class SimpleFFN( MLP ):
    class Factory( MLPFactory ):
        @staticmethod
        def getInstance( Properties: PropertiesManager ):
            return SimpleFFN( Properties )

    def __init__( self, Properties: PropertiesManager ):
        super( SimpleFFN, self ).__init__( Properties )


    def build_mlp_model(self, input_dim, nb_classes):
        Model = Sequential()
        #input layer
        Model.add(
            Dense(
                units=1000,
                input_dim = input_dim,
            )
        )
        #hidden layer
        Model.add(
            Dense(
                units = 500,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "sigmoid",
            )
        )
        #output layer
        Model.add( Dense( units = nb_classes, activation ='softmax' ) )

        Model.compile(
            loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy']
        )

        Model.summary()
        self._Model = Model
