from pandas import Series
from numpy import array as Array
from biomed.vectorizer.vectorizer import Vectorizer
from biomed.vectorizer.vectorizer import VectorizerFactory
from biomed.vectorizer.selector.selector import Selector
from biomed.properties_manager import PropertiesManager
from biomed.services_getter import ServiceGetter
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import float64

class StdVectorizer( Vectorizer ):
    def __init__( self, Properties: PropertiesManager, Selector: Selector ):
        self.__Properties = Properties
        self.__Selector = Selector
        self.__Vectorizer = None

    def __initializeVectorizer( self ):
         self.__Vectorizer = TfidfVectorizer(
             analyzer = self.__Properties.vectorizing[ 'analyzer' ],
             min_df = self.__Properties.vectorizing[ 'min_df' ],
             max_df = self.__Properties.vectorizing[ 'max_df' ],
             max_features = self.__Properties.vectorizing[ 'max_features' ],
             ngram_range = self.__Properties.vectorizing[ 'ngram_range' ],
             use_idf = self.__Properties.vectorizing[ 'use_idf' ],
             smooth_idf = self.__Properties.vectorizing[ 'smooth_idf' ],
             sublinear_tf = self.__Properties.vectorizing[ 'sublinear_tf' ],
             norm = self.__Properties.vectorizing[ 'norm' ],
             binary = self.__Properties.vectorizing[ 'binary' ],
             dtype = float64,
        )

    def featureizeTrain( self, Train: Series, Labels: Series ) -> Array:
        self.__initializeVectorizer()
        Features = self.__Vectorizer.fit_transform( Train )
        self.__Selector.build( Features, Labels )
        return self.__Selector.select( Features )

    def __checkVectorizer( self ):
        if not self.__Vectorizer:
            raise RuntimeError( "You must extract trainings feature, before" )

    def featureizeTest( self, Test: Series ) -> Array:
        self.__checkVectorizer()
        return self.__Selector.select(
            self.__Vectorizer.transform( Test )
        )

    def getSupportedFeatures( self ) -> list:
        self.__checkVectorizer()
        return self.__Selector.getSupportedFeatures(
            self.__Vectorizer.get_feature_names()
        )

    class Factory( VectorizerFactory ):
        @staticmethod
        def getInstance( getService: ServiceGetter ):
            return StdVectorizer(
                getService( 'properties', PropertiesManager ),
                getService( 'vectorizer.selector', Selector ),
            )
