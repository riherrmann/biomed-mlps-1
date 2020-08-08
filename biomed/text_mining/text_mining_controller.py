from biomed.text_mining.controller import Controller, ControllerFactory
from biomed.properties_manager import PropertiesManager
from biomed.facilitymanager.facility_manager import FacilityManager
from biomed.splitter.splitter import Splitter
from biomed.preprocessor.preprocessor import Preprocessor
from biomed.vectorizer.vectorizer import Vectorizer
from biomed.mlp.mlp import MLP
from biomed.evaluator.evaluator import Evaluator
from biomed.mlp.input_data import InputData
from biomed.services_getter import ServiceGetter
from pandas import DataFrame, Series
from tensorflow.keras.utils import to_categorical as hotEncode
from numpy import array as Array

class TextminingController( Controller ):
    def __init__(
        self,
        Properties: PropertiesManager,
        FacilityManager: FacilityManager,
        Splitter: Splitter,
        Preprocessor: Preprocessor,
        Vectorizer: Vectorizer,
        MLP: MLP,
        Evaluator: Evaluator
    ):
        self.__Properties = Properties
        self.__FacilityManager = FacilityManager
        self.__Splitter = Splitter
        self.__Evaluator = Evaluator
        self.__Preprocessor = Preprocessor
        self.__Vectorizer = Vectorizer
        self.__Model = MLP

        self.__Data = None
        self.__Categories = None

    def __splitIntoTestAndTrainingData( self ) -> list:
        return self.__Splitter.trainingSplit(
            self.__Data[ 'pmid' ],
            self.__Data[ self.__Properties.classifier ]
        )

    def __mapIdsToKey( self, Ids: Series, Key: str ) -> Series:
        Set = self.__Data[ Key ]
        Set.index = list( self.__Data[ 'pmid' ] )
        return Set.filter( items = list( Ids ) )

    def __splitIntoValidationAndTrainingData( self, TrainingIds: Series ) -> tuple:
        return self.__Splitter.validationSplit(
            TrainingIds,
            self.__mapIdsToKey( TrainingIds, self.__Properties.classifier )
        )

    def __runFolds( self ):
        Folds = self.__splitIntoTestAndTrainingData()
        Index = 1
        for Fold in Folds:
            print( "run fold #{}".format( str( Index ) ) )
            if 1 < len( Folds ):
                self.__Evaluator.setFold( Index )

            self.__runFold( Fold[ 0 ], Fold[ 1 ] )

            Index += 1

    def __preprocess( self ) -> Series:
        print( "preprocessing....." )

        Corpus = self.__Preprocessor.preprocessCorpus(
            self.__Data[ 'pmid' ],
            self.__Data[ 'text' ]
        )

        self.__Evaluator.capturePreprocessingTime()
        self.__Evaluator.capturePreprocessedData(
            Corpus,
            self.__Data[ 'text' ]
        )

        return Corpus

    def __vectorize(
        self,
        Corpus: Series,
        TrainingIds: Series,
        TestIds: Series
    ) -> tuple:
        print( "vectorizing..." )

        Labels = self.__Data[ self.__Properties.classifier ]
        Labels.index = self.__Data[ 'pmid' ]
        Labels = Labels.filter( TrainingIds )

        TrainingFeatures = self.__Vectorizer.featureizeTrain(
            Corpus.filter( list( TrainingIds ) ),
            Labels
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

    def __hotEncodeLabel( self, Ids: Series ) -> tuple:
        return hotEncode(
            self.__Data[ self.__Properties.classifier ]
                .filter( list( Ids ) )
                .to_numpy(),
            len( self.__Categories )
        )

    def __makeInputData( self, Training: tuple, Validation: tuple, Test: tuple ) -> tuple:
        Features = InputData(
            Training[ 1 ],
            Validation[ 1 ],
            Test[ 1 ]
        )

        Labels = InputData(
            self.__hotEncodeLabel( Training[ 0 ] ),
            self.__hotEncodeLabel( Validation[ 0 ] ),
            self.__hotEncodeLabel( Test[ 0 ] ),
        )

        return ( Features, Labels )

    def __train( self, Features: InputData, Labels: InputData ):
        print( "training...." )
        print( "Training: {}\nValidation: {}\nTest: {}".format(
            Features.Training.shape,
            Features.Validation.shape,
            Features.Test.shape
        ) )
        Structure = self.__Model.buildModel( Features.Training.shape[ 1 ] )
        History = self.__Model.train( Features, Labels )
        Score = self.__Model.getTrainingScore( Features, Labels )

        self.__Evaluator.captureModel( Structure )
        self.__Evaluator.captureTrainingTime()
        self.__Evaluator.captureTrainingHistory( History )
        self.__Evaluator.captureEvaluationScore( Score )

    def __predict( self, TestIds: Series, Features: InputData, Labels: InputData ):
        print( "prediciting..." )
        Predictions = self.__Model.predict( Features.Test )
        Expected = list( self.__Data[ self.__Properties.classifier ].filter( TestIds ) )

        self.__Evaluator.caputrePredictingTime()
        self.__Evaluator.capturePredictions(
            Predictions,
            TestIds,
            Expected
        )

        self.__Evaluator.score(
            Predictions,
            Expected,
            self.__Categories
    )

    def __trainAndPredict( self, Training: tuple, Test: tuple ):
        TestIds = Test[ 0 ]
        Training, Validation = self.__validationSplit( Training )
        Features, Labels = self.__makeInputData( Training, Validation, Test )
        self.__train( Features, Labels )
        self.__predict( TestIds, Features, Labels )

    def __printResults( self, Results: dict ):
        print( 'f1 score:' )
        print( Results[ 'score' ] )

        print( 'details:' )
        print( Results[ 'report' ] )

    def __runFold(
        self,
        TrainingIds: Series,
        TestIds: Series,
    ):
        self.__Evaluator.captureData( TrainingIds, TestIds )
        self.__Evaluator.captureStartTime()
        TrainingFeatures, TestFeatures = self.__vectorize(
            self.__preprocess(),
            TrainingIds,
            TestIds
        )

        self.__trainAndPredict(
            ( TrainingIds, TrainingFeatures ),
            ( TestIds, TestFeatures )
        )

        self.__printResults( self.__Evaluator.finalize() )

    def __getCategories( self ):
        self.__Categories = self.__Data[ self.__Properties.classifier ].unique()
        self.__Categories.sort()

    def process(
        self,
        Data: DataFrame,
        TestData: None,
        ShortName: str,
        Description: str
    ):
        self.__Evaluator.start( ShortName, Description )
        self.__Data = self.__FacilityManager.clean( Data )
        self.__getCategories()
        self.__runFolds()

    class Factory( ControllerFactory ):
        @staticmethod
        def getInstance( getService: ServiceGetter ) -> Controller:
            return TextminingController(
                getService( 'properties', PropertiesManager ),
                getService( 'facilitymanager', FacilityManager ),
                getService( 'splitter', Splitter ),
                getService( 'preprocessor', Preprocessor ),
                getService( 'vectorizer', Vectorizer ),
                getService( 'mlp', MLP ),
                getService( 'evaluator', Evaluator )
            )
