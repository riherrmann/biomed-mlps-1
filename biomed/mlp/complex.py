from keras.models import Sequential
from keras.layers import Dense, Dropout
from biomed.properties_manager import PropertiesManager
from keras.regularizers import l1
from biomed.mlp.mlp import MLP
from biomed.mlp.mlp import MLPFactory

class ComplexFFN( MLP ):
    class Factory( MLPFactory ):
        @staticmethod
        def getInstance( Properties: PropertiesManager ):
            return ComplexFFN( Properties )

    def __init__( self, Properties: PropertiesManager ):
        super( ComplexFFN, self ).__init__( Properties )


    def build_mlp_model(self, input_dim, nb_classes):
        Model = Sequential()
        #input layer
        #hidden layers
        Model.add(
            Dense(
                units=1024,
                input_dim = input_dim,
                activation = "relu",
                activity_regularizer = l1(0.0001)
            )
        )
        #hidden layers
        Model.add(
            Dense(
                units=512,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "relu",
            )
        )
        Model.add( Dropout( 0.5 ) )
        Model.add(
            Dense(
                units = 256,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "relu",
            )
        )
        Model.add( Dropout( 0.45 ) )
        Model.add(
            Dense(
                units = 128,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "relu",
            )
        )
        Model.add( Dropout( 0.4 ) )
        Model.add(
            Dense(
                units = 128,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "relu",
            )
        )
        Model.add( Dropout( 0.35 ) )
        Model.add(
            Dense(
                units = 64,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "relu",
            )
        )
        Model.add( Dropout( 0.3 ) )
        Model.add(
            Dense(
                units = 32,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "relu",
            )
        )
        Model.add( Dropout( 0.25 ) )
        Model.add(
            Dense(
                units = 16,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "relu",
            )
        )
        Model.add( Dropout( 0.2 ) )
        Model.add(
            Dense(
                units = 8,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "relu",
            )
        )
        Model.add( Dropout( 0.15 ) )
        Model.add(
            Dense(
                units = 4,
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros",
                activation = "relu",
            )
        )
        Model.add( Dropout( 0.1 ) )
        #output layer
        Model.add( Dense( units = nb_classes, activation ='sigmoid' ) )

        Model.compile(
            loss='mean_squared_error',
            optimizer='sgd',
            metrics=['accuracy']
        )

        Model.summary()
        self._Model = Model
