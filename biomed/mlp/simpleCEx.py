from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1
from biomed.properties_manager import PropertiesManager
from biomed.mlp.model_base import ModelBase

class SimpleCExtendedFFN( ModelBase ):
    def __init__( self, Properties: PropertiesManager ):
        super( SimpleCExtendedFFN, self ).__init__( Properties )

    def buildModel( self, Dimensions: int ) -> str:
        Model = Sequential()
        #input layer
        Model.add(
            Dense(
                units = 10,
                input_dim = Dimensions,
                activity_regularizer= l1( 0.0001 ),
                activation = "relu",
            )
        )
        #hidden layer
        Model.add( Dropout( 0.25 ) )
        Model.add(
            Dense(
                units = 20,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "relu"
            )
        )
        #output layer
        Model.add( Dropout( 0.1 ) )
        Model.add( Dense( units = 2, activation ='sigmoid' ) )

        Model.compile(
            loss="binary_crossentropy",
            optimizer="sgd",
            metrics=['accuracy']
        )

        self._Model = Model
        return self._summarize()
