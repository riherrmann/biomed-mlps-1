from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
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
        Model.add(
            Dense(
                units = 5,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "relu"
            )
        )
        #output layer
        Model.add( Dense( units = nb_classes, activation ='sigmoid' ) )

        Rate = 0.1
        Decay = Rate / self._Properties.training_properties[ 'epochs' ]
        Momentum = 0.8

        Sgd = SGD(lr = Rate, momentum = Momentum, decay = Decay, nesterov = False )

        Model.compile(
            loss="binary_crossentropy",
            optimizer=Sgd,
            metrics=['accuracy']
        )

        Model.summary()
        self._Model = Model
