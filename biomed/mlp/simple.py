from keras.models import Sequential
from keras.layers import Dense, Dropout
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
        Model.add(
            Dense(
                units=1000,
                input_dim = input_dim,
                activation = "sigmoid",
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros"
            )
        )
        Model.add( Dropout( 0.30 ) )
        Model.add(
            Dense(
                units = 500,
                activation = "sigmoid",
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros"
            )
        )
        Model.add( Dropout( 0.25 ) )
        Model.add(
            Dense(
                units = 300,
                activation="sigmoid",
                kernel_initializer="random_uniform",
                bias_initializer="zeros"
            )
        )
        Model.add( Dropout( 0.20 ) )
        Model.add(
            Dense(
                units = 200,
                activation = "sigmoid",
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros"
            )
        )
        Model.add( Dropout( 0.15 ) )
        Model.add(
            Dense(
                units = 50,
                activation = "sigmoid",
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros"
            )
        )
        Model.add( Dropout( 0.10 ) )
        Model.add(
            Dense(
                units = 25,
                activation = "sigmoid",
                kernel_initializer = "random_uniform",
                bias_initializer = "zeros"
            )
        )
        Model.add( Dropout( 0.05 ) )
        Model.add(
            Dense(
                units = nb_classes,
                activation = "sigmoid"
            )
        )

        Model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        Model.summary()
        self._Model = Model
