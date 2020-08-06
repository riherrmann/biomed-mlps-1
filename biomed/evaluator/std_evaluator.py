from biomed.evaluator.evaluator import Evaluator
from biomed.evaluator.evaluator import EvaluatorFactory
from biomed.properties_manager import PropertiesManager
from biomed.utils.dir_checker import checkDir, toAbsPath
from biomed.utils.file_writer import FileWriter
import biomed.services as Services
from numpy import array as Array
from pandas import Series, DataFrame
import os as OS
from os import mkdir
from datetime import datetime as Time
from sys import getsizeof as memSize
from sklearn.metrics import f1_score as F1
from sklearn.metrics import classification_report as Reporter
import asyncio

class StdEvaluator( Evaluator ):
    def __init__(
        self,
        Properties: PropertiesManager,
        SimpleWriter: FileWriter,
        JSONWriter: FileWriter,
        CSVWriter: FileWriter
    ):
        self.__Properties = Properties
        self.__Path = None
        self.__SimpleWriter = SimpleWriter
        self.__JSONWriter = JSONWriter
        self.__CSVWriter = CSVWriter
        self.__Time = dict()
        self.__LastScore = None
        self.__LastReport = None
        self.__Model = None
        self.__Steps = []

    def __setPath( self, ShortName: str ):
        self.__Path = OS.path.join(
            self.__Properties.result_dir,
            "{}-{}".format( ShortName, Time.now().strftime( '%Y-%m-%d_%H-%M-%S' ) )
        )

    def __makePathForFile( self, FileName: str ) -> str:
        return OS.path.join( self.__Path, FileName )

    def __writeJSON( self, FileName: str, Content: dict ):
        self.__JSONWriter.write(
            self.__makePathForFile( FileName ),
            Content
        )

    def __writeFile( self, FileName: str, Content: list ):
        self.__SimpleWriter.write(
            self.__makePathForFile( FileName ),
            Content
        )

    def __writeCSV( self, FileName: str, Content: dict ):
        self.__CSVWriter.write(
            self.__makePathForFile( FileName ),
            Content
        )

    def __makeFrameAndSave(
        self,
        FileName: str,
        Data,
        Columns: list,
        Rows: list = None
    ):
        if not Rows:
            DF = DataFrame(
                Data,
                columns = Columns
            )
        else:
            DF = DataFrame(
                Data,
                columns = Columns,
                index = Rows
            )

        DF.to_csv( self.__makePathForFile( FileName ) )

    def start( self, ShortName: str, Desription: str ):
        checkDir( toAbsPath( self.__Properties.result_dir ) )
        self.__setPath( ShortName )
        mkdir( self.__Path )
        self.__writeJSON( 'config.json', self.__Properties.toDict() )
        self.__writeFile( 'descr.txt', Desription.strip().splitlines() )

    def setFold( self, Fold ):
        self.__checkIfIsStarted()
        self.__Path = OS.path.join( self.__Path, str( Fold ) )
        mkdir( self.__Path )

    def __checkIfIsStarted( self ):
        if not self.__Path:
            raise RuntimeError( "You have to start the Evaluator before caputuring stuff" )

    def __enqueueStep( self, Step ):
        self.__checkIfIsStarted()
        self.__Steps.append( Step )

    def captureStartTime( self ):
        self.__Time[ 'start' ] = int( Time.now().strftime( '%s' ) )

    def capturePreprocessingTime( self ):
        self.__Time[ 'preprocessing' ] = int( Time.now().strftime( '%s' ) )

    def captureVectorizingTime( self ):
        self.__Time[ 'vectorizing' ] = int( Time.now().strftime( '%s' ) )

    def captureTrainingTime( self ):
        self.__Time[ 'training' ] = int( Time.now().strftime( '%s' ) )

    def caputrePredictingTime( self ):
        self.__Time[ 'predicting' ] = int( Time.now().strftime( '%s' ) )

    async def __captureData( self, Train: Series, Test: Series ):
        self.__writeCSV( 'train.csv', { 'pmid': list( Train ) } )
        self.__writeCSV( 'test.csv', { 'pmid': list( Test ) } )

    def captureData( self, Train: Series, Test: Series ):
        self.__enqueueStep( self.__captureData( Train, Test ) )

    async def __capturePreprocessedData( self, Processed: Series, Original: Series ):
        Sizes = {
            'processed': memSize( list( Processed ) ),
            'original': memSize( list( Original ) )
        }

        self.__writeCSV( "sizes.csv", Sizes )

    def capturePreprocessedData( self, TrainDocs: Series, TestDocs: Series ):
        self.__enqueueStep( self.__capturePreprocessedData( TrainDocs, TestDocs ) )

    async def __captureFeatures(
        self,
        TrainFeatures: tuple,
        TestFeatures: tuple,
        BagOfWords: list
    ):
        self.__makeFrameAndSave(
            'trainingFeatures.csv',
            TrainFeatures[ 1 ],
            BagOfWords,
            list( TrainFeatures[ 0 ] )
        )

        self.__makeFrameAndSave(
            'testFeatures.csv',
            TestFeatures[ 1 ],
            BagOfWords,
            list( TestFeatures[ 0 ] )
        )

    def captureFeatures(
        self,
        TrainFeatures: tuple,
        TestFeatures: tuple,
        BagOfWords: list
    ):
        self.__enqueueStep(
            self.__captureFeatures( TrainFeatures, TestFeatures, BagOfWords )
        )

    async def __captureModel( self, Model: str ):
        self.__writeFile(
            'model.txt',
            Model.strip().splitlines()
        )

    def captureModel( self, Model: str ):
        self.__Model = Model
        self.__enqueueStep( self.__captureModel( Model ) )

    async def __captureTrainingHistory( self, History: dict ):
        self.__writeCSV(
            'trainingHistory.csv',
            History
        )

    def captureTrainingHistory( self, History: dict ):
        self.__enqueueStep( self.__captureTrainingHistory( History ) )

    async def __captureEvaluationScore( self, Score: dict ):
        self.__writeCSV(
            'evalScore.csv',
            Score
        )

    def captureEvaluationScore( self, Score: dict ):
        self.__enqueueStep( self.__captureEvaluationScore( Score ) )

    def __justSavePredictions( self, Predictions: Array, PMIds: list ):
        self.__makeFrameAndSave(
            'predictions.csv',
            [ PMIds, list( Predictions ) ],
            [ 'pmid', self.__Properties.classifier ]
        )

    def __saveLabeledPredictions( self, Predictions: Array, PMIds: list, Labels: list ):
        self.__makeFrameAndSave(
            'predictions.csv',
            [ list( Predictions ), Labels ],
            [ 'predicted', 'actual' ],
            PMIds
        )

    async def __capturePredictions( self, Predictions: Array, PMIds: Series, Actual: Series = None ):
        if not Actual:
            self.__justSavePredictions( Predictions, list( PMIds ) )
        else:
            self.__saveLabeledPredictions( Predictions, list( PMIds ), list( Actual ) )

    def capturePredictions(
        self,
        Predictions: Array,
        PMIds: Series,
        Actual: Series = None
    ):
        self.__enqueueStep(
            self.__capturePredictions( Predictions, PMIds, Actual )
        )

    def __getMacroAndMicroScore( self, Predicted: Array, Actual: list ) -> tuple:
        return (
            F1(
                y_pred = Predicted,
                y_true = Actual,
                average = 'macro'
            ),
            F1(
                y_pred = Predicted,
                y_true = Actual,
                average = 'micro'
            )
        )

    def __scoreBinary( self, Predictions: Array, Actual: list ) -> list:
        MiMa = self.__getMacroAndMicroScore( Predictions, Actual )
        Score = [
            MiMa[ 0 ],
            MiMa[ 1 ],
            F1(
                y_pred = Predictions,
                y_true = Actual,
                average = 'binary'
            )
        ]

        self.__makeFrameAndSave(
            'f1.csv',
            Score,
            [ 'macro', 'micro', 'binary' ],
        )

        return Score

    def __scoreMulitClass( self, Predictions: Array, Actual: list ) -> list:
        MiMa = self.__getMacroAndMicroScore( Predictions, Actual )
        Score = [
            MiMa[ 0 ],
            MiMa[ 1 ],
            F1(
                y_pred = Predictions,
                y_true = Actual,
                average = 'samples'
            )
        ]

        self.__makeFrameAndSave(
            'f1.csv',
            Score,
            [ 'macro', 'micro', 'samples' ],
        )

        return Score

    def __makeReport( self, Predicted: Array, Actual: list, Labels: list ) -> str:
        self.__writeCSV(
            'classReport.csv',
            Reporter(
                y_pred = Predicted,
                y_true = Actual,
                labels = Labels,
                output_dict = True
            )
        )

        return Reporter(
            y_pred = Predicted,
            y_true = Actual,
            labels = Labels,
            output_dict = False
        )

    async def __score(
        self,
        Predictions: Array,
        Actual: Series,
        Labels: Series
    ):
        Actual = list( Actual )
        Labels = list( Labels )

        if len( Labels ) == 2:
            self.__LastScore = self.__scoreBinary( Predictions, Actual )
        else:
            self.__LastScore = self.__scoreMulitClass( Predictions, Actual )

        self.__LastReport = self.__makeReport( Predictions, Actual, Labels )

    def score( self, Predictions: Array, Actual: Series, Labels: Series ):
        self.__enqueueStep( self.__score( Predictions, Actual, Labels ) )

    async def __waitForSteps( self ):
        while self.__Steps:
            await self.__Steps.pop()

    def __renderResults( self ) -> dict:
        Results = {}

        if self.__Model:
            Results[ 'model' ] = self.__Model

        if self.__LastScore:
            Results[ 'score' ] = self.__LastScore

        if self.__LastReport:
            Results[ 'report' ] = self.__LastReport

        if self.__Time:
            Results[ 'time' ] = self.__Time

        return Results

    def finalize( self ) -> dict:
        self.__checkIfIsStarted()
        self.__writeCSV( 'time.csv', self.__Time )
        asyncio.get_event_loop().run_until_complete( self.__waitForSteps() )
        return self.__renderResults()

    class Factory( EvaluatorFactory ):
        @staticmethod
        def getInstance() -> Evaluator:
            return StdEvaluator(
                Properties = Services.getService( 'properties', PropertiesManager ),
                SimpleWriter = Services.getService( 'evaluator.simple', FileWriter ),
                JSONWriter = Services.getService( 'evaluator.json', FileWriter ),
                CSVWriter = Services.getService( 'evaluator.csv', FileWriter ),
            )
