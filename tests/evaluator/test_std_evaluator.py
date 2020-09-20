import unittest
from unittest.mock import patch, MagicMock, ANY
from biomed.evaluator.evaluator import Evaluator
from biomed.evaluator.std_evaluator import StdEvaluator
from biomed.properties_manager import PropertiesManager
from biomed.utils.file_writer import FileWriter
from datetime import datetime
from pandas import Series, DataFrame
import os as OS
from numpy import array as Array

class StdEvaluatorSpec( unittest.TestCase ):
    def setUp( self ):
        self.__PM = PropertiesManager()
        self.__Simple = MagicMock( spec = FileWriter )
        self.__JSON = MagicMock( spec = FileWriter )
        self.__CSV = MagicMock( spec = FileWriter )

        self.__mkdirM = patch( 'biomed.evaluator.std_evaluator.mkdir' )
        self.__mkdir = self.__mkdirM.start()
        self.__checkDirM = patch( 'biomed.evaluator.std_evaluator.checkDir' )
        self.__checkDir = self.__checkDirM.start()
        self.__checkDir.return_value = True
        self.__TimeM = patch( 'biomed.evaluator.std_evaluator.Time' )
        self.__Time = self.__TimeM.start()
        self.__TimeObj = MagicMock( spec = datetime )
        self.__Time.now.return_value = self.__TimeObj
        self.__TimeValue = '2020-07-25_14-53-36'
        self.__TimeObj.strftime.return_value = self.__TimeValue

    def tearDown( self ):
        self.__mkdirM.stop()
        self.__checkDirM.stop()
        self.__TimeM.stop()

    def __fakeLocator( self, ServiceKey: str, __ ):
        Dependencies = {
            'evaluator.simple': self.__Simple,
            'evaluator.json': self.__JSON,
            'evaluator.csv': self.__CSV,
            'properties': self.__PM
        }

        return Dependencies[ ServiceKey ]

    def test_it_is_a_evluator( self ):
        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        self.assertTrue( isinstance( MyEval, Evaluator ) )

    def test_it_depends_on_properties_and_all_FileWriter( self ):
        Dependencies = {
            'evaluator.simple': FileWriter,
            'evaluator.json': FileWriter,
            'evaluator.csv': FileWriter,
            'properties': PropertiesManager
        }

        def fakeLocator( ServiceKey, Type ):
            if ServiceKey not in Dependencies:
                raise RuntimeError( "Unexpected ServiceKey" )

            if Type != Dependencies[ ServiceKey ]:
                raise RuntimeError( "Unexpected Type" )

            return PropertiesManager()

        ServiceGetter = MagicMock()
        ServiceGetter.side_effect = fakeLocator

        StdEvaluator.Factory.getInstance( ServiceGetter )
        self.assertEqual(
            4,
            ServiceGetter.call_count
        )

        self.assertEqual(
            len( Dependencies.keys() ),
            ServiceGetter.call_count
        )

    def test_it_makes_a_dir_to_store_the_data( self ):
        ShortName = "test"

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test of the module" )

        self.__mkdir.assert_called_once_with(
            OS.path.join(
                self.__PM.result_dir,
                '{}-{}'.format( ShortName, self.__TimeValue )
            )
        )

    def test_it_writes_the_config_into_a_json( self ):
        ShortName = "test"
        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue )
        )

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test of the module" )

        self.__JSON.write.assert_called_once_with(
            OS.path.join( Path, 'config.json' ),
            self.__PM.toDict()
        )

    def test_it_writes_the_description_into_a_txt( self ):
        ShortName = "test"
        Description = "test of the module\n"

        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue ),
        )

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, Description )

        self.__Simple.write.assert_any_call(
            OS.path.join( Path, 'descr.txt' ),
            [ Description.strip() ]
        )

        self.__Simple.reset_mock()

        Description = "test2 of the module\nwith multilines"
        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, Description )

        self.__Simple.write.assert_any_call(
            OS.path.join( Path, 'descr.txt' ),
            [ 'test2 of the module', 'with multilines' ]
        )

    def test_it_fails_if_the_evaluator_is_not_started_while_caputure_the_splitted_data( self ):
        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        with self.assertRaises( RuntimeError, msg = "You have to start the Evaluator before caputuring stuff" ):
            MyEval.captureData( MagicMock(), MagicMock() )
            MyEval.finalize()

    def test_it_captures_the_pmids_of_trainings_and_test_data( self ):
        ShortName = "test"
        Train = Series( [ 12, 123, 423, 21 ] )
        Test = Series( [ 32, 42, 23 ] )

        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue )
        )

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test run" )
        MyEval.captureData( Train, Test )
        MyEval.finalize()

        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'train.csv' ),
            { 'pmid': list( Train ) }
        )
        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'test.csv' ),
            { 'pmid': list( Test ) }
        )

    def test_it_fails_if_the_evaluator_is_not_started_while_caputure_the_class_weights( self ):
        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        with self.assertRaises( RuntimeError, msg = "You have to start the Evaluator before caputuring stuff" ):
            MyEval.captureClassWeights( MagicMock() )
            MyEval.finalize()


    def test_it_captures_the_class_weights_if_there_are_present( self ):
        ShortName = "test"
        Weights = Series( [ 0.23, 0.1, 0.42 ] )
        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue )
        )

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, MagicMock() )
        MyEval.captureClassWeights( Weights )
        MyEval.finalize()

        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'weights.csv' ),
            { 'weights': list( Weights ) }
        )

        self.assertEqual(
            2,
            self.__CSV.write.call_count
        )

    def test_it_writes_no_class_weights_if_there_are_not_present( self ):
        Weights = None

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( MagicMock(), MagicMock() )
        MyEval.captureClassWeights( Weights )
        MyEval.finalize()

        self.assertEqual(
            1,
            self.__CSV.write.call_count
        )

    def test_it_fails_if_the_evaluator_is_not_started_while_caputure_the_processed_data( self ):
        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        with self.assertRaises( RuntimeError, msg = "You have to start the Evaluator before caputuring stuff" ):
            MyEval.capturePreprocessedData( MagicMock(), MagicMock() )
            MyEval.finalize()

    def test_it_caputures_the_size_of_the_documents_before_and_after_preprocessing(
        self,
    ):
        ShortName = "Test"
        Org = MagicMock( spec = Series )
        Pro = MagicMock( spec = Series )

        OrgSize = 42
        ProSize = 23

        Org.memory_usage.return_value = OrgSize
        Pro.memory_usage.return_value = ProSize


        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue )
        )

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test run" )
        MyEval.capturePreprocessedData( Pro, Org )
        MyEval.finalize()

        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'sizes.csv' ),
            { 'processed': ProSize, 'original': OrgSize }
        )

        Org.memory_usage.assert_called_once_with( deep = True )
        Pro.memory_usage.assert_called_once_with( deep = True )

    @patch( 'biomed.evaluator.std_evaluator.DataFrame' )
    def test_it_fails_if_the_evaluator_is_not_started_while_caputure_the_features(
        self,
        DF: MagicMock
    ):
        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        with self.assertRaises( RuntimeError, msg = "You have to start the Evaluator before caputuring stuff" ):
            MyEval.captureFeatures( MagicMock(), MagicMock(), MagicMock() )
            MyEval.finalize()

    @patch( 'biomed.evaluator.std_evaluator.DataFrame' )
    def test_it_captures_the_trainings_and_test_features(
        self,
        DF: MagicMock
    ):
        TrainIds = [ 123, 3, 53343 ]
        TrainingsFeatures = [ [1, 2,], [5, 6,], [9, 10,] ] # this should be a array
        TestIds = [ 23, 42 ]
        TestFeatures = [ [3,4,], [7,8], [11, 0] ] # this should be a array

        BagOfWords = [ 'a', 'b' ]

        ShortName = "Test"

        Frame = MagicMock( spec = DataFrame )
        DF.return_value = Frame

        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue )
        )

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test run" )
        MyEval.captureFeatures(
            ( Series( TrainIds ), TrainingsFeatures ),
            ( Series( TestIds ), TestFeatures ),
            BagOfWords
        )
        MyEval.finalize()

        DF.assert_any_call(
            TrainingsFeatures,
            columns = BagOfWords,
            index = list( TrainIds )
        )

        Frame.to_csv.assert_any_call( OS.path.join( Path, 'trainingFeatures.csv' ) )

        DF.assert_any_call(
            TestFeatures,
            columns = BagOfWords,
            index = list( TestIds )
        )

        Frame.to_csv.assert_any_call( OS.path.join( Path, 'testFeatures.csv' ) )

    def test_it_fails_if_the_evaluator_is_not_started_while_caputure_the_training_history( self ):
        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        with self.assertRaises( RuntimeError, msg = "You have to start the Evaluator before caputuring stuff" ):
            MyEval.captureTrainingHistory( MagicMock() )
            MyEval.finalize()

    def test_it_caputures_the_training_history( self ):
        History = { 'accuracy': [ 0 ], 'loss': [ 1 ] }
        ShortName = "Test"

        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue )
        )

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test run" )
        MyEval.captureTrainingHistory( History )
        MyEval.finalize()

        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'trainingHistory.csv' ),
            History
        )

    def test_it_fails_if_the_evaluator_is_not_started_while_caputure_the_evaluation_score( self ):
        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        with self.assertRaises( RuntimeError, msg = "You have to start the Evaluator before caputuring stuff" ):
            MyEval.captureEvaluationScore( MagicMock() )
            MyEval.finalize()

    def test_it_caputures_the_evaluation_score( self ):
        Score = { 'accuracy': 0, 'loss': 1 }
        ShortName = "Test"

        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue )
        )

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test run" )
        MyEval.captureEvaluationScore( Score )
        MyEval.finalize()

        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'evalScore.csv' ),
            Score
        )

    def test_it_fails_if_the_evaluator_is_not_started_while_caputure_the_predictions( self ):
        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        with self.assertRaises( RuntimeError, msg = "You have to start the Evaluator before caputuring stuff" ):
            MyEval.capturePredictions( MagicMock(), MagicMock(), MagicMock() )
            MyEval.finalize()

    @patch( 'biomed.evaluator.std_evaluator.DataFrame' )
    def test_it_saves_the_predictions_and_eventually_their_corresponding_labels(
        self,
        DF: DataFrame
    ):
        ShortName = "Test"

        Frame = MagicMock( spec = DataFrame )
        Ids = [ 1, 2, 3, 4 ]
        Predicted = Array( [ 1, 0, 1, 0 ] )
        Actual = [ 1, 1, 1, 0 ]

        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue )
        )

        DF.return_value = Frame

        self.__PM.classifier = 'is_cancer'

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test run" )
        MyEval.capturePredictions( Predicted, Ids )
        MyEval.finalize()

        DF.assert_any_call(
            { 'pmid': Ids, self.__PM.classifier: Predicted.tolist() },
            columns = [ 'pmid', self.__PM.classifier ],
        )

        Frame.to_csv.assert_any_call( OS.path.join( Path, 'predictions.csv' ) )
        Frame.reset_mock()

        self.__PM.classifier = 'doid'
        MyEval.capturePredictions( Predicted, Ids )
        MyEval.finalize()

        DF.assert_any_call(
            { 'pmid': Ids, self.__PM.classifier: Predicted.tolist() },
            columns = [ 'pmid', self.__PM.classifier ],
        )

        Frame.to_csv.assert_any_call( OS.path.join( Path, 'predictions.csv' ) )
        Frame.reset_mock()

        MyEval.capturePredictions( Predicted, Ids, Actual )
        MyEval.finalize()

        DF.assert_any_call(
            { 'predicted': Predicted.tolist(), 'actual': Actual },
            columns = [ 'predicted', 'actual' ],
            index = Ids
        )

        Frame.to_csv.assert_any_call( OS.path.join( Path, 'predictions.csv' ) )

    def test_it_fails_if_the_evaluator_is_not_started_while_scoring( self ):
        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        with self.assertRaises( RuntimeError, msg = "You have to start the Evaluator before caputuring stuff" ):
            MyEval.score( MagicMock(), MagicMock(), MagicMock() )
            MyEval.finalize()


    @patch( 'biomed.evaluator.std_evaluator.Reporter' )
    @patch( 'biomed.evaluator.std_evaluator.F1' )
    @patch( 'biomed.evaluator.std_evaluator.DataFrame' )
    def test_it_scores_and_saves_the_predictions_for_binary(
        self,
        DF: MagicMock,
        Scorer: MagicMock,
        _
    ):
        ShortName = "Test"

        Frame = MagicMock( spec = DataFrame )
        Predicted = [ 1, 0, 1, 0 ] #this should be a array
        Actual = [ 1, 1, 1, 0 ]
        Labels = [ 0, 1 ]

        Micro = 0.23
        Macro = 0.5
        Binary = 0.42

        def fakeScore( y_pred, y_true, average ):
            if average == 'micro':
                return Micro
            elif average == 'macro':
                return Macro
            else:
                return Binary

        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue )
        )

        Scorer.side_effect = fakeScore
        DF.return_value = Frame

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test run" )
        MyEval.score( Predicted, Series( Actual ), Series( Labels ) )
        MyEval.finalize()

        Scorer.assert_any_call(
            y_pred = Predicted,
            y_true = Actual,
            average = 'macro'
        )

        Scorer.assert_any_call(
            y_pred = Predicted,
            y_true = Actual,
            average = 'micro'
        )

        Scorer.assert_any_call(
            y_pred = Predicted,
            y_true = Actual,
            average = 'binary'
        )

        DF.assert_any_call(
            [ [ Macro, Micro, Binary ] ],
            columns = [ 'macro', 'micro', 'binary' ],
        )

        Frame.to_csv.assert_any_call( OS.path.join( Path, 'f1.csv' ) )

    @patch( 'biomed.evaluator.std_evaluator.Reporter' )
    @patch( 'biomed.evaluator.std_evaluator.F1' )
    @patch( 'biomed.evaluator.std_evaluator.DataFrame' )
    def test_it_scores_and_saves_the_predictions_for_multi_class(
        self,
        DF: MagicMock,
        Scorer: MagicMock,
        _
    ):
        ShortName = "Test"

        Frame = MagicMock( spec = DataFrame )
        Predicted = [ 1, 2, 0, 0 ] #this should be a array
        Actual = [ 1, 2, 1, 0 ]
        Labels = [ 0, 1, 2 ]

        Micro = 0.23
        Macro = 0.5
        Sample = 0.42

        def fakeScore( y_pred, y_true, average ):
            if average == 'micro':
                return Micro
            elif average == 'macro':
                return Macro
            else:
                return Sample

        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue )
        )

        Scorer.side_effect = fakeScore
        DF.return_value = Frame

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test run" )
        MyEval.score( Predicted, Series( Actual ), Series( Labels ) )
        MyEval.finalize()

        Scorer.assert_any_call(
            y_pred = Predicted,
            y_true = Actual,
            average = 'macro'
        )

        Scorer.assert_any_call(
            y_pred = Predicted,
            y_true = Actual,
            average = 'micro'
        )

        Scorer.assert_any_call(
            y_pred = Predicted,
            y_true = Actual,
            average = 'weighted'
        )

        DF.assert_any_call(
            [ [ Macro, Micro, Sample ] ],
            columns = [ 'macro', 'micro', 'weighted' ],
        )

        Frame.to_csv.assert_any_call( OS.path.join( Path, 'f1.csv' ) )


    @patch( 'biomed.evaluator.std_evaluator.Reporter' )
    @patch( 'biomed.evaluator.std_evaluator.F1' )
    @patch( 'biomed.evaluator.std_evaluator.DataFrame' )
    def test_it_saves_the_classification_report(
        self,
        _,
        __,
        Reporter: MagicMock
    ):
        ShortName = "Test"

        Predicted = [ 1, 2, 0, 0 ] #this should be a array
        Actual = [ 1, 2, 1, 0 ]
        Labels = [ 0, 1, 2 ]

        Report = { 'asd': { 'ls': 'weq', 'weq': 'qwe' } }

        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue )
        )

        Reporter.return_value = Report

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test run" )
        MyEval.score( Predicted, Series( Actual ), Series( Labels ) )
        MyEval.finalize()

        Reporter.assert_any_call(
            y_pred = Predicted,
            y_true = Actual,
            labels = Labels,
            output_dict = True
        )

        Reporter.assert_any_call(
            y_pred = Predicted,
            y_true = Actual,
            labels = Labels,
            output_dict = False
        )

        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'classReport.csv' ),
            Report
        )

    def test_it_fails_if_the_evaluator_is_not_started_while_finializing( self ):
        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        with self.assertRaises( RuntimeError, msg = "You have to start the Evaluator before caputuring stuff" ):
            MyEval.finalize()

    def test_it_writes_the_time_metrix_while_finalizing( self ):
        Times = [ self.__TimeValue, '1', '2', '3', '4', '5' ]
        TimeIndex = [ -1 ]
        ShortName = "Test"

        def fakeTimes( _ ):
            TimeIndex[ 0 ] += 1
            return Times[ TimeIndex[ 0 ] ]

        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue )
        )

        self.__TimeObj.strftime.side_effect = fakeTimes

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test run" )
        MyEval.captureStartTime()
        MyEval.capturePreprocessingTime()
        MyEval.captureVectorizingTime()
        MyEval.captureTrainingTime()
        MyEval.caputrePredictingTime()
        MyEval.finalize()

        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'time.csv' ),
            { 'start': 1, 'preprocessing': 2, 'vectorizing': 3, 'training': 4, 'predicting': 5 }
        )

    @patch( 'biomed.evaluator.std_evaluator.Reporter' )
    @patch( 'biomed.evaluator.std_evaluator.F1' )
    @patch( 'biomed.evaluator.std_evaluator.DataFrame' )
    def test_it_waits_for_various_results_while_finalizing( self, DF: MagicMock, _, __ ):
        ShortName = "Test"

        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue )
        )

        Frame = MagicMock( spec = DataFrame )
        DF.return_value = Frame

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test run" )
        MyEval.captureData( MagicMock(), MagicMock() )
        MyEval.captureClassWeights( MagicMock( spec = Series ) )
        MyEval.capturePreprocessedData( MagicMock(), MagicMock() )
        MyEval.captureFeatures( MagicMock(), MagicMock(), MagicMock() )
        MyEval.captureTrainingHistory( MagicMock() )
        MyEval.captureEvaluationScore( MagicMock() )
        MyEval.capturePredictions( MagicMock(), MagicMock() )
        MyEval.score( MagicMock(), MagicMock(), MagicMock() )
        MyEval.finalize()

        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'train.csv' ),
            ANY
        )
        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'test.csv' ),
            ANY
        )
        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'weights.csv' ),
            ANY
        )
        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'sizes.csv' ),
            ANY
        )

        Frame.to_csv.assert_any_call( OS.path.join( Path, 'trainingFeatures.csv' ) )
        Frame.to_csv.assert_any_call( OS.path.join( Path, 'testFeatures.csv' ) )

        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'trainingHistory.csv' ),
            ANY
        )

        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'evalScore.csv' ),
            ANY
        )

        Frame.to_csv.assert_any_call( OS.path.join( Path, 'predictions.csv' ) )
        Frame.to_csv.assert_any_call( OS.path.join( Path, 'f1.csv' ) )
        self.__CSV.write.assert_any_call(
            OS.path.join( Path, 'classReport.csv' ),
            ANY
        )

    @patch( 'biomed.evaluator.std_evaluator.Reporter' )
    @patch( 'biomed.evaluator.std_evaluator.F1' )
    @patch( 'biomed.evaluator.std_evaluator.DataFrame' )
    def test_it_return_a_dict_with_given_results( self, DF: MagicMock, F1: MagicMock, Reporter: MagicMock ):
        ShortName = "Test"

        Model = MagicMock()
        Score = MagicMock()
        Report = MagicMock()

        Frame = MagicMock( spec = DataFrame )
        DF.return_value = Frame
        F1.return_value = Score
        Reporter.return_value = Report

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test run" )
        MyEval.captureData( MagicMock(), MagicMock() )
        MyEval.capturePreprocessedData( MagicMock(), MagicMock() )
        MyEval.captureFeatures( MagicMock(), MagicMock(), MagicMock() )
        MyEval.captureModel( Model )
        MyEval.captureTrainingHistory( MagicMock() )
        MyEval.captureEvaluationScore( MagicMock() )
        MyEval.capturePredictions( MagicMock(), MagicMock() )
        MyEval.score( MagicMock(), MagicMock(), MagicMock() )
        self.assertDictEqual(
            MyEval.finalize(),
            {
                'model': Model,
                'score': [ Score, Score, Score ],
                'report': Report,
            }
        )

    def test_it_fails_if_the_evaluator_is_not_started_while_caputure_the_model( self ):
        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        with self.assertRaises( RuntimeError, msg = "You have to start the Evaluator before caputuring stuff" ):
            MyEval.captureModel( MagicMock() )
            MyEval.finalize()

    def test_it_captures_the_model( self ):
        ShortName = "test"
        Model = "I will be the model str\nmulitlined\n"

        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue ),
        )

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "ANY" )
        MyEval.captureModel( Model )
        MyEval.finalize()

        self.__Simple.write.assert_any_call(
            OS.path.join( Path, 'model.txt' ),
            Model.strip().splitlines()
        )

    def test_it_fails_if_the_evaluator_is_not_started_while_setting_a_fold( self ):
        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        with self.assertRaises( RuntimeError, msg = "You have to start the Evaluator before caputuring stuff" ):
            MyEval.setFold( 1 )

    def test_it_makes_a_sub_dir_for_each_fold( self ):
        ShortName = "test"
        Fold = 1

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test of the module" )
        MyEval.setFold( Fold )

        self.__mkdir.assert_any_call(
            OS.path.join(
                self.__PM.result_dir,
                '{}-{}'.format( ShortName, self.__TimeValue ),
                str( Fold )
            )
        )

        Fold = 2

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "test of the module" )
        MyEval.setFold( Fold )

        self.__mkdir.assert_any_call(
            OS.path.join(
                self.__PM.result_dir,
                '{}-{}'.format( ShortName, self.__TimeValue ),
                str( Fold )
            )
        )

    def test_it_writes_data_in_fold_dir( self ):
        ShortName = "test"
        Model = "I will be the model str\nmulitlined\n"

        Fold = 1
        Path = OS.path.join(
            self.__PM.result_dir,
            '{}-{}'.format( ShortName, self.__TimeValue ),
            str( Fold )
        )

        MyEval = StdEvaluator.Factory.getInstance( self.__fakeLocator )
        MyEval.start( ShortName, "ANY" )
        MyEval.setFold( Fold )
        MyEval.captureModel( Model )
        MyEval.finalize()

        self.__Simple.write.assert_any_call(
            OS.path.join( Path, 'model.txt' ),
            Model.strip().splitlines()
        )
