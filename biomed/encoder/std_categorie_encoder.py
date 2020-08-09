from biomed.encoder.categorie_encoder import CategoriesEncoder, CategoriesEncoderFactory
from pandas import Series
from sklearn.preprocessing import LabelEncoder as Encoder
from tensorflow.keras.utils import to_categorical as hotEncode
from numpy import array as Array

class StdCategoriesEncoder( CategoriesEncoder ):
    def __init__( self ):
        self.__Encoder = None

    def setCategories( self, Categories: Series ):
        Categories = Categories.unique()
        Categories.sort()

        self.__Encoder = Encoder()
        self.__Encoder.fit( Categories )

    def __verifyFitted( self ):
        if not self.__Encoder:
            raise RuntimeError( 'No Categories had been fit in so far' )

    def getCategories( self ) -> list:
        self.__verifyFitted()
        return self.__Encoder.classes_.tolist()

    def amountOfCategories( self ):
        self.__verifyFitted()
        return self.__Encoder.classes_.size

    def encode( self, ToEncode: Array ) -> Array:
        self.__verifyFitted()
        return self.__Encoder.transform( ToEncode )

    def hotEncode( self, ToEncode: Array ) -> Array:
        self.__verifyFitted()
        return hotEncode(
            self.encode( ToEncode ),
            self.amountOfCategories()
        )

    def decode( self, ToDecode: Array ) -> Array:
        self.__verifyFitted()
        return self.__Encoder.inverse_transform( ToDecode )

    class Factory( CategoriesEncoderFactory ):
        def getInstance() -> CategoriesEncoder:
            return StdCategoriesEncoder()
