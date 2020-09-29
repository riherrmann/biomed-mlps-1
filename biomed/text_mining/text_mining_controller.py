from biomed.text_mining.controller import Controller, ControllerFactory
from biomed.properties_manager import PropertiesManager
from biomed.facilitymanager.facility_manager import FacilityManager
from biomed.splitter.splitter import Splitter
from biomed.preprocessor.preprocessor import Preprocessor
from biomed.vectorizer.vectorizer import Vectorizer
from biomed.measurer.measurer import Measurer
from biomed.mlp.mlp import MLP
from biomed.evaluator.evaluator import Evaluator
from biomed.mlp.input_data import InputData
from biomed.encoder.categorie_encoder import CategoriesEncoder
from biomed.services_getter import ServiceGetter
from pandas import DataFrame, Series
from typing import Union
from numpy import array as Array, unique

class TextminingController( Controller ):
    def __init__(
        self,
        Properties: PropertiesManager,
        Encoder: CategoriesEncoder,
        FacilityManager: FacilityManager,
        Splitter: Splitter,
        Preprocessor: Preprocessor,
        Vectorizer: Vectorizer,
        Measurer: Measurer,
        MLP: MLP,
        Evaluator: Evaluator
    ):
        self.__Properties = Properties
        self.__Encoder = Encoder
        self.__FacilityManager = FacilityManager
        self.__Splitter = Splitter
        self.__Evaluator = Evaluator
        self.__Preprocessor = Preprocessor
        self.__Vectorizer = Vectorizer
        self.__Measurer = Measurer
        self.__Model = MLP

        self.__Prefix = 'bce1f597-48e3-42e8-a140-1675e60f34f3-'
        self.__Data = None
        self.__TestData = None
        self.__Categories = None

    def __isInProduction( self ):
        return isinstance( self.__TestData, DataFrame )

    def __mapIdsToKey( self, Ids: Series, Key: str ) -> Series:
        Set = self.__Data[ Key ]
        Set.index = list( self.__Data[ 'pmid' ] )
        return Set.filter( items = list( Ids ) )

    def __splitIntoValidationAndTrainingData( self, TrainingIds: Series ) -> tuple:
        return self.__Splitter.validationSplit(
            TrainingIds,
            self.__mapIdsToKey( TrainingIds, self.__Properties.classifier )
        )

    def __mergeSeries( self, S1: Series, I1: Series, S2: Series, I2: Series ) -> Series:
         return Series(
             list( S1 ) + list( S2 ),
             index = list( I1 ) + list( I2 ),
         )

    def __preprocessTrainingsData( self ) -> Series:
         return self.__Preprocessor.preprocessCorpus(
            self.__Data[ 'pmid' ],
            self.__Data[ 'text' ]
        )

    def __preprocessTesTData( self ) -> Series:
        return self.__Preprocessor.preprocessCorpus(
            self.__TestData[ 'pmid' ],
            self.__TestData[ 'text' ]
        )

    def __preprocessThemAll( self ) -> Series:
        if self.__isInProduction():
            return self.__mergeSeries(
                self.__preprocessTrainingsData(),
                self.__Data[ 'pmid' ],
                self.__preprocessTesTData(),
                self.__TestData[ 'pmid' ],
            )
        else:
            return self.__preprocessTrainingsData()

    def __getOrgText( self ):
        if self.__isInProduction():
            return self.__mergeSeries(
                self.__Data[ 'text' ],
                self.__Data[ 'pmid' ],
                self.__TestData[ 'text' ],
                self.__TestData[ 'pmid' ],
            )
        else:
            return self.__Data[ 'text' ]

    def __preprocess( self ) -> Series:
        print( "preprocessing....." )

        Corpus = self.__preprocessThemAll()

        self.__Evaluator.capturePreprocessingTime()
        self.__Evaluator.capturePreprocessedData(
            Corpus,
            self.__getOrgText()
        )

        return Corpus

    def __vectorize(
        self,
        Corpus: Series,
        TrainingIds: Series,
        TestIds: Series,
        Weights: Union[ None, dict ],
    ) -> tuple:
        print( "vectorizing..." )

        Labels = self.__Data[ self.__Properties.classifier ]
        Labels.index = self.__Data[ 'pmid' ]
        Labels = Labels.filter( TrainingIds )

        TrainingFeatures = self.__Vectorizer.featureizeTrain(
            Corpus.filter( list( TrainingIds ) ),
            Labels,
            Weights,
        )

        TestFeatures = self.__Vectorizer.featureizeTest(
            Corpus.filter( list( TestIds ) )
        )

        self.__Evaluator.captureVectorizingTime()
        self.__Evaluator.captureFeatures(
            ( TrainingIds, TrainingFeatures ),
            ( TestIds, TestFeatures ),
            self.__Vectorizer.getSupportedFeatures()
        )

        return ( TrainingFeatures, TestFeatures )

    def __convertToArray( self, Value: Series ):
        return Array( list( Value ) )

    def __validationSplit(
        self,
        Training: tuple
    ) -> tuple:
        TrainingIds, TrainingFeatures = Training
        Training, Validation = self.__Splitter.validationSplit(
            TrainingIds,
            self.__mapIdsToKey( TrainingIds, self.__Properties.classifier )
        )

        Features = Series( TrainingFeatures.tolist() )
        Features.index = list( TrainingIds )

        TrainingFeatures = Features.filter( list( Training ) )
        ValidationFeatures = Features.filter( list( Validation ) )

        return (
            ( Training, self.__convertToArray( TrainingFeatures ) ),
            ( Validation, self.__convertToArray( ValidationFeatures ) )
        )

    def __hotEncodeLabels( self, Ids: Series ) -> tuple:
        return self.__Encoder.hotEncode(
            self.__convertToArray(
                list( self.__Data[ self.__Properties.classifier ].filter( list( Ids ) ) )
            )
        )

    def __hotEncodeTestLabels( self, TestIds: Series ) -> Union[ tuple, None ]:
        if not self.__isInProduction():
            return self.__hotEncodeLabels( TestIds )
        else:
            return None

    def __makeInputData( self, Training: tuple, Validation: tuple, Test: tuple ) -> tuple:
        Features = InputData(
            Training[ 1 ],
            Validation[ 1 ],
            Test[ 1 ]
        )

        Labels = InputData(
            self.__hotEncodeLabels( Training[ 0 ] ),
            self.__hotEncodeLabels( Validation[ 0 ] ),
            self.__hotEncodeTestLabels( Test[ 0 ] ),
        )

        return ( Features, Labels )

    def __evaluateModel( self, Features: InputData, Labels: InputData ) -> None:
        if not self.__isInProduction():
            self.__Evaluator.captureEvaluationScore(
                self.__Model.getTrainingScore( Features, Labels )
            )

    def __train( self, Features: InputData, Labels: InputData, Weights: Union[ None, dict ] ):
        print( "training...." )
        print( "Training: {}\nValidation: {}\nTest: {}".format(
            Features.Training.shape,
            Features.Validation.shape,
            Features.Test.shape
        ) )

        Structure = self.__Model.buildModel( Features.Training.shape, Weights )
        History = self.__Model.train( Features, Labels )

        self.__Evaluator.captureModel( Structure )
        self.__Evaluator.captureTrainingTime()
        self.__Evaluator.captureTrainingHistory( History )

        self.__evaluateModel( Features, Labels )

    def __getExpectedTestLabels( self, TestIds ) -> list:
        if self.__isInProduction():
            return None
        else:
            return list( self.__Data[ self.__Properties.classifier ].filter( TestIds ) )

    def __score( self, Predictions: Array, Expected: Union[ None, Array ] ) -> None:
        if not self.__isInProduction():
            self.__Evaluator.score(
                Predictions,
                Expected,
                self.__Encoder.getCategories()
            )

    #TODO make this better
    def __cleanTestIds( self, TestIds: Series ) -> Series:
        if self.__isInProduction():
            TestIds = list( TestIds )
            for Index in range( 0, len( TestIds ) ):
                TestIds[ Index ] = TestIds[ Index ].lstrip( self.__Prefix )

            TestIds = Series( TestIds )

        return TestIds

    def __predict( self, TestIds: Series, Features: InputData, Labels: InputData ):
        print( "prediciting..." )

        Predictions = self.__Model.predict( Features.Test )
        Predictions = self.__Encoder.decode( Predictions )
        Expected = self.__getExpectedTestLabels( TestIds )

        self.__Evaluator.caputrePredictingTime()
        self.__Evaluator.capturePredictions(
            Predictions,
            self.__cleanTestIds( TestIds ),
            Expected
        )

        self.__score( Predictions, Expected )

    def __trainAndPredict( self, Training: tuple, Test: tuple, Weights: Union[ None, dict ] ):
        TestIds = Test[ 0 ]

        Training, Validation = self.__validationSplit( Training )
        Features, Labels = self.__makeInputData( Training, Validation, Test )

        self.__train( Features, Labels, Weights )
        self.__predict( TestIds, Features, Labels )

    def __printResults( self, Results: dict ):
        print( 'f1 score:' )
        if 'score' in Results:
            print( Results[ 'score' ] )

        print( 'details:' )
        if 'report' in Results:
            print( Results[ 'report' ] )

    def __runFold(
        self,
        TrainingIds: Series,
        TestIds: Series,
    ):
        self.__Evaluator.captureData( TrainingIds, TestIds )
        self.__Evaluator.captureStartTime()

        Actual = self.__convertToArray(
            self.__mapIdsToKey( TrainingIds, self.__Properties.classifier )
        )

        Weights = self.__Measurer.measureClassWeights(
            unique( Actual ),
            Actual
        )

        self.__Evaluator.captureClassWeights( Weights )
        TrainingFeatures, TestFeatures = self.__vectorize(
            self.__preprocess(),
            TrainingIds,
            TestIds,
            Weights,
        )

        self.__trainAndPredict(
            ( TrainingIds, TrainingFeatures ),
            ( TestIds, TestFeatures ),
            Weights,
        )

        self.__printResults( self.__Evaluator.finalize() )

    def __splitIntoTestAndTrainingData( self ) -> list:
        return self.__Splitter.trainingSplit(
            self.__Data[ 'pmid' ],
            self.__Data[ self.__Properties.classifier ]
        )

    def __splitOrReflect( self ) -> list:
        if self.__isInProduction():
            return [ ( self.__Data[ 'pmid' ], self.__TestData[ 'pmid' ] ) ]
        else:
            return self.__splitIntoTestAndTrainingData()


    def __runFolds( self ):
        Folds = self.__splitOrReflect()
        Index = 1
        for Fold in Folds:
            print( "run fold #{}".format( str( Index ) ) )
            if 1 < len( Folds ):
                self.__Evaluator.setFold( Index )

            self.__runFold( Fold[ 0 ], Fold[ 1 ] )

            Index += 1

    def __prefixTestIds( self, TestData: Union[ None, DataFrame ] ) -> Union[ None, DataFrame ]:
        if isinstance( TestData, DataFrame ):
            Ids = [ '{}{}'.format( self.__Prefix, Id ) for Id in list( TestData[ 'pmid' ] ) ]
            TestData.pmid = Ids

            return TestData
        else:
            return None

    def process(
        self,
        Data: DataFrame,
        TestData: Union[ None, DataFrame ],
        ShortName: str,
        Description: str
    ):
        self.__Evaluator.start( ShortName, Description )

        self.__Data = self.__FacilityManager.clean( Data )
        self.__TestData = self.__prefixTestIds( TestData )

        self.__Encoder.setCategories( self.__Data[ self.__Properties.classifier ] )
        self.__runFolds()

    class Factory( ControllerFactory ):
        @staticmethod
        def getInstance( getService: ServiceGetter ) -> Controller:
            return TextminingController(
                getService( 'properties', PropertiesManager ),
                getService( 'categories', CategoriesEncoder ),
                getService( 'facilitymanager', FacilityManager ),
                getService( 'splitter', Splitter ),
                getService( 'preprocessor', Preprocessor ),
                getService( 'vectorizer', Vectorizer ),
                getService( 'measurer', Measurer ),
                getService( 'mlp', MLP ),
                getService( 'evaluator', Evaluator )
            )
