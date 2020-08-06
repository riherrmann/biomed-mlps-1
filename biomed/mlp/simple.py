from keras.models import Sequential
from keras.layers import Dense
from biomed.properties_manager import PropertiesManager
from keras.regularizers import l1
from biomed.mlp.model_base import ModelBase

class SimpleFFN( ModelBase ):
    def __init__( self, Properties: PropertiesManager ):
        super( SimpleFFN, self ).__init__( Properties )

    def buildModel( self, Dimensions: int ) -> str:
        Model = Sequential()
        #input layer
        Model.add(
            Dense(
                units=10,
                input_dim = Dimensions,
                activation = "relu",
                activity_regularizer = l1( 0.0001 )
            )
        )
        #hidden layer
        Model.add(
            Dense(
                units = 5,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "relu",
            )
        )
        #output layer
        Model.add( Dense( units = 2, activation ='sigmoid' ) )

        Model.compile(
            loss='mean_squared_error',
            optimizer='sgd',
            metrics=['accuracy']
        )

        self._Model = Model
        return self._summarize()
