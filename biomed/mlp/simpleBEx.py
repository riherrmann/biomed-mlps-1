from keras.models import Sequential
from keras.layers import Dense, Dopout
from keras.regularizers import l1
from keras.losses import BinaryCrossentropy
from biomed.properties_manager import PropertiesManager
from biomed.mlp.mlp import MLP
from biomed.mlp.mlp import MLPFactory

class SimpleBExtendedFFN( MLP ):
    class Factory( MLPFactory ):
        @staticmethod
        def getInstance( Properties: PropertiesManager ):
            return SimpleBExtendedFFN( Properties )

    def __init__( self, Properties: PropertiesManager ):
        super( SimpleBExtendedFFN, self ).__init__( Properties )


    def build_mlp_model(self, input_dim, nb_classes):
        Model = Sequential()
        #input layer
        Model.add(
            Dense(
                units = 10,
                input_dim = input_dim,
                activity_regularizer= l1( 0.0001 ),
                activation = "relu",
            )
        )
        #hidden layer
        Model.add( Dopout( 0.25 ) )
        Model.add(
            Dense(
                units = 5,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "relu"
            )
        )
        #output layer
        Model.add( Dopout( 0.1 ) )
        Model.add( Dense( units = nb_classes, activation ='sigmoid' ) )

        Model.compile(
            loss="binary_crossentropy",
            optimizer="sgd",
            metrics=['accuracy']
        )

        Model.summary()
        self._Model = Model
