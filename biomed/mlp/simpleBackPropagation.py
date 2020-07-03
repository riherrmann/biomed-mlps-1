import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Layer
from biomed.properties_manager import PropertiesManager
from biomed.mlp.mlp import MLP
from biomed.mlp.mlp import MLPFactory

class BackPropagation( Layer ):
    def __init__( self, units=32, target="is_cancer"):
        super( BackPropagation, self ).__init__()
        self.w = self.add_weight(
            shape=( self.__dimensions( target ), units ),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            shape=(units,),
            initializer="zeros",
            trainable=True
        )

    def __dimensions( self, Target ):
        if Target == "is_cancer":
            return 2
        else:
            return 42 #TODO

    def call( self, inputs ):
        return tf.matmul( inputs, self.w ) + self.b

class SimpleBackPropagationFFN( MLP ):
    class Factory( MLPFactory ):
        @staticmethod
        def getInstance( Properties: PropertiesManager ):
            return SimpleBackPropagationFFN( Properties )

    def __init__( self, Properties: PropertiesManager ):
        super( SimpleBackPropagationFFN, self ).__init__( Properties )


    def build_mlp_model(self, input_dim, nb_classes):
        Model = Sequential()
        Model.add(
            Dense(
                units=1000,
                input_dim = input_dim,
                activation = "sigmoid",
                kernel_initializer = "glorot_uniform",
                bias_initializer = "zeros"
            )
        )
        Model.add( Dropout( 0.30 ) )
        Model.add(
            Dense(
                units = 500,
                activation = "sigmoid",
                kernel_initializer = "glorot_uniform",
                bias_initializer = "zeros"
            )
        )
        Model.add( Dropout( 0.25 ) )
        Model.add(
            Dense(
                units = 300,
                activation="sigmoid",
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros"
            )
        )
        Model.add( Dropout( 0.20 ) )
        Model.add(
            Dense(
                units = 200,
                activation = "sigmoid",
                kernel_initializer = "glorot_uniform",
                bias_initializer = "zeros"
            )
        )
        Model.add( Dropout( 0.15 ) )
        Model.add(
            Dense(
                units = 50,
                activation = "sigmoid",
                kernel_initializer = "glorot_uniform",
                bias_initializer = "zeros"
            )
        )
        Model.add( Dropout( 0.10 ) )
        Model.add(
            Dense(
                units = 25,
                activation = "sigmoid",
                kernel_initializer = "glorot_uniform",
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
        Model.add(
            BackPropagation(
                units = nb_classes,
            ),
        )
        Model.add( Activation( 'softmax' ) )

        Model.compile(
            loss='mean_squared_error',
            optimizer='sgd',
            metrics=['accuracy']
        )

        Model.summary()
        self._Model = Model
