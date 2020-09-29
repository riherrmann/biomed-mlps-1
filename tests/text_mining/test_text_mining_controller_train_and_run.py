import unittest
from unittest.mock import patch, MagicMock, ANY
from biomed.text_mining.controller import Controller
from biomed.text_mining.text_mining_controller import TextminingController
from biomed.properties_manager import PropertiesManager
from biomed.facilitymanager.facility_manager import FacilityManager
from biomed.splitter.splitter import Splitter
from biomed.preprocessor.preprocessor import Preprocessor
from biomed.vectorizer.vectorizer import Vectorizer
from biomed.measurer.measurer import Measurer
from biomed.mlp.mlp import MLP
from biomed.mlp.input_data import InputData
from biomed.evaluator.evaluator import Evaluator
from biomed.encoder.categorie_encoder import CategoriesEncoder
from pandas import DataFrame, Series
from numpy import array as Array

class TextminingControllerSpec( unittest.TestCase ):
    def setUp( self ):
        self.__Data = DataFrame(
            {
                'pmid': [ '1a', '2a', '3a', '4a' ],
                'text': [
                    "My little cute Poney is a Poney",
                    "My little farm is cute.",
                    "My little programm is a application and runs and runs and runs.",
                    "My little keyboard is to small"
                ],
                'is_cancer': [ 0, 1, 1, 0 ],
                'doid': [ -1, 1, 2, -1 ],
                'cancer_type': [ 'no cancer', 'cancer', 'cancer', 'no cancer' ],
            },
            columns = [ 'pmid', 'cancer_type', 'doid', 'is_cancer', 'text' ]
        )

        self.__TestData = DataFrame(
            {
                'pmid': [ '1b', '2b', '3b', '4b' ],
                'text': [
                    "My little cute Poney is a Poney!",
                    "My little farm is cute?",
                    "My little programm is a application and runs and runs and runs...",
                    "My little keyboard is to small :("
                ],
            },
            columns = [ 'pmid', 'text' ]
        )

        self.__INDF = patch( 'biomed.text_mining.text_mining_controller.InputData' )

        self.__PM = PropertiesManager()
        self.__Encoder = MagicMock( spec = CategoriesEncoder )
        self.__FacilityManager = MagicMock( spec = FacilityManager )
        self.__Splitter = MagicMock( spec = Splitter )
        self.__Preprocessor = MagicMock( spec = Preprocessor )
        self.__Vectorizer = MagicMock( spec = Vectorizer )
        self.__Measurer = MagicMock( spec = Measurer )
        self.__MLP = MagicMock( spec = MLP )
        self.__Evaluator = MagicMock( spec = Evaluator )
        self.__IND = self.__INDF.start()
        self.__IND.Training = MagicMock()
        self.__IND.Training.shape = ( MagicMock(), MagicMock() )
        self.__IND.Validation = MagicMock()
        self.__IND.Test = MagicMock()

        self.__FacilityManager.clean.return_value = self.__Data

        self.__Splitter.trainingSplit.return_value = [ ( MagicMock(), MagicMock() ) ]
        self.__Splitter.validationSplit.return_value = ( MagicMock(), MagicMock() )

        def reflectCorpus( X:Series, Y: Series ) -> Series:
            return Series( list( Y ), index = list( X ) )

        self.__Preprocessor.preprocessCorpus.side_effect = reflectCorpus

        TrainFeatures = MagicMock()
        TrainFeatures.tolist.return_value = [ [1],[2],[3],[4] ]

        self.__Vectorizer.featureizeTrain.return_value = TrainFeatures
        self.__Vectorizer.featureizeTest.return_value = MagicMock()
        self.__Vectorizer.getSupportedFeatures.return_value = MagicMock()
        self.__Measurer.measureClassWeights.return_value = MagicMock()

        self.__Merged = Series(
            list( self.__Data[ 'text' ] ) + list( self.__TestData[ 'text' ] ),
            index = list( self.__Data[ 'pmid' ] ) + list( self.__TestData[ 'pmid' ] )
        )

        self.__Prefix = 'bce1f597-48e3-42e8-a140-1675e60f34f3'

    def tearDown( self ):
        self.__INDF.stop()

    def __fakeLocator( self, ServiceKey: str, _ ):
        Dependencies = {
            'properties': self.__PM,
            'categories': self.__Encoder,
            'facilitymanager': self.__FacilityManager,
            'splitter': self.__Splitter,
            'preprocessor': self.__Preprocessor,
            'vectorizer': self.__Vectorizer,
            'measurer': self.__Measurer,
            'mlp': self.__MLP,
            'evaluator': self.__Evaluator
        }

        return Dependencies[ ServiceKey ]

    def test_it_is_a_Controller( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        self.assertTrue( isinstance( MyController, Controller ) )

    def test_it_depends_on_many_things( self ):
        def fakeLocator( ServiceKey: str, Type ):
            Dependencies = {
                'properties': PropertiesManager,
                'categories': CategoriesEncoder,
                'splitter': Splitter,
                'facilitymanager': FacilityManager,
                'preprocessor': Preprocessor,
                'vectorizer': Vectorizer,
                'measurer':Measurer,
                'mlp': MLP,
                'evaluator': Evaluator
            }

            if not ServiceKey in Dependencies.keys():
                raise RuntimeError( 'Unknown ServiceKey {}'.format( ServiceKey ) )

            if not Type == Dependencies[ ServiceKey ]:
                raise RuntimeError( 'Unknown Type for {}'.format( ServiceKey ) )

            return MagicMock( spec = Dependencies[ ServiceKey ] )

        ServiceGetter = MagicMock()
        ServiceGetter.side_effect = fakeLocator

        TextminingController.Factory.getInstance( ServiceGetter )
        self.assertEqual(
            9,
            ServiceGetter.call_count
        )

    def test_it_starts_the_evaluator( self ):
        Name = MagicMock()
        Description = MagicMock()

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = Name,
            Description = Description
        )

        self.__Evaluator.start.assert_called_once_with(
            Name,
            Description
        )

    def test_it_cleans_up_the_Data( self ):
        Data = MagicMock()

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__FacilityManager.clean.assert_called_once_with( Data )

    def test_it_weight_classes_for_binary( self ):
        self.__PM.classifier = 'is_cancer'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Classes, CurrentLabels = self.__Measurer.measureClassWeights.call_args_list[ 0 ][ 0 ]

        self.__Measurer.measureClassWeights.assert_called_once()

        self.assertListEqual(
            [ 0, 1 ],
            Classes.tolist()
        )

        self.assertListEqual(
            list( self.__Data[ self.__PM.classifier ] ),
            CurrentLabels.tolist(),
        )

    def test_it_weight_classes_for_multiclass( self ):
        self.__PM.classifier = 'doid'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Classes, CurrentLabels = self.__Measurer.measureClassWeights.call_args_list[ 0 ][ 0 ]

        self.__Measurer.measureClassWeights.assert_called_once()

        self.assertListEqual(
            [ -1, 1, 2 ],
            Classes.tolist()
        )

        self.assertListEqual(
            list( self.__Data[ self.__PM.classifier ] ),
            CurrentLabels.tolist(),
        )

    def test_it_captures_the_class_weights( self ):
        Weights = MagicMock()

        self.__Measurer.measureClassWeights.return_value = Weights

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureClassWeights.assert_called_once_with( Weights )

    def test_it_sets_the_categories_for_binary( self ):
        self.__PM.classifier = 'is_cancer'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Encoder.setCategories.assert_called_once_with(
            self.__Data[ 'is_cancer' ]
        )

    def test_it_sets_the_categories_for_mulit_class( self ):
        self.__PM.classifier = 'doid'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Encoder.setCategories.assert_called_once_with(
            self.__Data[ 'doid' ]
        )

    def test_it_splits_not_the_test_data_for_binary( self ):
        self.__PM.classifier = 'is_cancer'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Splitter.trainingSplit.assert_not_called()

    def test_it_splits_the_test_data_for_mulitclass( self ):
        self.__PM.classifier = 'doid'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Splitter.trainingSplit.assert_not_called()

    def test_it_caputures_the_ids_of_training_including_the_validation_ids_and_test_sets( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureData.assert_called_once_with(
            self.__Data[ 'pmid' ],
            self.__TestData[ 'pmid' ]
        )

    def test_it_caputure_the_start_time( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureStartTime.assert_called_once()

    def test_it_preprocesses_the_trainings_data_including_validation_data( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Preprocessor.preprocessCorpus.assert_any_call(
            self.__Data[ 'pmid' ],
            self.__Data[ 'text' ]
        )

    def test_it_preprocesses_the_test_data( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Preprocessor.preprocessCorpus.assert_called_with(
            self.__TestData[ 'pmid' ],
            self.__TestData[ 'text' ]
        )

    def test_it_captures_the_preprocessing_time( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.capturePreprocessingTime.assert_called_once()

    def test_it_captures_the_preprocessed_coprus_in_compare_to_org_corpus( self ):
        ProCorpus = MagicMock()

        self.__Preprocessor.preprocessCorpus.return_value = ProCorpus

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.capturePreprocessedData(
            Processed = ProCorpus,
            Original = self.__Merged
        )

    def test_it_vectorizes_the_trainings_coprus_for_binary( self ):
        Labels = self.__Data[ 'is_cancer' ]
        Weights = MagicMock()

        TrainFeatures = MagicMock()
        TrainFeatures.tolist.return_value = [ [1],[2],[3],[4] ]

        self.__Vectorizer.featureizeTrain.return_value = TrainFeatures
        self.__PM.classifier = 'is_cancer'
        self.__Measurer.measureClassWeights.return_value = Weights

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Vectorizer.featureizeTrain.call_args_list[ 0 ]

        self.assertListEqual(
            list( self.__Data[ 'text' ] ),
            list( Arguments[ 0 ] )
        )

        self.assertListEqual(
            list( Labels ),
            list( Arguments[ 1 ] )
        )

        self.assertEqual(
            Weights,
            Arguments[ 2 ]
        )

        self.__Vectorizer.featureizeTrain.assert_called_once()

    def test_it_vectorizes_the_trainings_coprus_for_mulitclass( self ):
        Weights = MagicMock()

        Labels = self.__Data[ 'doid' ]

        TrainFeatures = MagicMock()
        TrainFeatures.tolist.return_value = [ [1],[2],[3],[4] ]
        self.__Vectorizer.featureizeTrain.return_value = TrainFeatures
        self.__PM.classifier = 'doid'
        self.__Measurer.measureClassWeights.return_value = Weights

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Vectorizer.featureizeTrain.call_args_list[ 0 ]

        self.assertListEqual(
            list( self.__Data[ 'text' ] ),
            list( Arguments[ 0 ] )
        )

        self.assertListEqual(
            list( Labels ),
            list( Arguments[ 1 ] )
        )

        self.assertEqual(
            Weights,
            Arguments[ 2 ]
        )

        self.__Vectorizer.featureizeTrain.assert_called_once()

    def test_it_vectorizes_the_test_coprus( self ):
        TrainFeatures = MagicMock()
        TrainFeatures.tolist.return_value = [ [1],[2],[3],[4] ]

        self.__Vectorizer.featureizeTrain.return_value = TrainFeatures

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Vectorizer.featureizeTest.call_args_list[ 0 ]

        self.assertListEqual(
            list( self.__TestData[ 'text' ] ),
            list( Arguments[ 0 ] )
        )

        self.__Vectorizer.featureizeTest.assert_called_once()

    def test_it_captures_the_vectorizing_time( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureVectorizingTime.assert_called_once()

    def test_it_captures_the_resulting_features_and_their_Labels( self ):
        TrainFeatures = MagicMock()
        TrainFeatures.tolist.return_value = [ [1],[2],[3],[4] ]
        TestFeatures = MagicMock()
        BagOfWords = MagicMock()
        self.__PM.evaluator[ 'captureFeatures' ] = True

        self.__Vectorizer.featureizeTrain.return_value = TrainFeatures
        self.__Vectorizer.featureizeTest.return_value = TestFeatures
        self.__Vectorizer.getSupportedFeatures.return_value = BagOfWords

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureFeatures.assert_called_once_with(
            ( self.__Data[ 'pmid' ], TrainFeatures ),
            ( self.__TestData[ 'pmid' ], TestFeatures ),
            BagOfWords
        )

    def test_it_splits_the_validation_data_of_the_trainings_data_for_binary( self ):
        TrainingFeatures = Array( [ [ 0., 2. ], [ 0.1, 0.3 ], [ 1.2, 4.3], [2.3, 4.2] ] )
        Expected = self.__Data[ 'is_cancer' ]
        Expected.index = list( self.__Data[ 'pmid' ] )

        self.__Vectorizer.featureizeTrain.return_value = TrainingFeatures

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Splitter.validationSplit.call_args_list[ 0 ]

        self.assertListEqual(
            list( self.__Data[ 'pmid' ] ),
            list( Arguments[ 0 ] )
        )

        self.assertListEqual(
            list( Expected ),
            list( Arguments[ 1 ] )
        )

        self.__Splitter.validationSplit.assert_called_once()

    def test_it_splits_the_validation_data_of_the_trainings_data_for_multiclass( self ):
        TrainingFeatures = Array( [ [ 0.1, 2.2 ], [ 0.13, 3.3 ], [ 1.25, 4.23], [ 2.1, 4.3 ] ] )
        Expected = self.__Data[ 'doid' ]
        Expected.index = list( self.__Data[ 'pmid' ] )

        self.__PM.classifier = 'doid'
        self.__Vectorizer.featureizeTrain.return_value = TrainingFeatures

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Splitter.validationSplit.call_args_list[ 0 ]

        self.assertListEqual(
            list( self.__Data[ 'pmid' ] ),
            list( Arguments[ 0 ] )
        )

        self.assertListEqual(
            list( Expected ),
            list( Arguments[ 1 ] )
        )

        self.__Splitter.validationSplit.assert_called_once()

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_brings_the_features_into_model_input_format(
        self,
        DataBinding: MagicMock
    ):
        TrainingFeatures = Array( [ [ 0., 2. ], [ 0.1, 0.3 ], [ 1.2, 4.3], [2.3, 4.2] ] )
        TestFeatures = Array( [ [ 0.1, 2.2 ], [ 0.13, 3.3 ], [ 1.25, 4.23], [ 2.1, 4.3 ] ] )
        ValidationIds = Series( [ '4a' ] )
        TrainingIds = Series( [ '1a', '2a', '3a' ] )

        self.__Splitter.validationSplit.return_value = ( TrainingIds, ValidationIds )
        self.__Vectorizer.featureizeTrain.return_value = TrainingFeatures
        self.__Vectorizer.featureizeTest.return_value = TestFeatures
        self.__PM.classifier = 'is_cancer'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        ArgumentsFeatures, _ = DataBinding.call_args_list[ 0 ]
        self.assertEqual(
            TrainingFeatures.tolist()[ 0:3 ],
            ArgumentsFeatures[ 0 ].tolist()
        )
        self.assertEqual(
            [ TrainingFeatures.tolist()[ 3 ] ],
            ArgumentsFeatures[ 1 ].tolist()
        )
        self.assertEqual(
            TestFeatures.tolist(),
            ArgumentsFeatures[ 2 ].tolist()
        )

    def test_it_hot_encodes_the_labels_for_binary( self ):
        TrainingFeatures = Array( [ [ 0., 2. ], [ 0.1, 0.3 ], [ 1.2, 4.3], [2.3, 4.2] ] )
        TestFeatures = Array( [ [ 0.1, 2.2 ], [ 0.13, 3.3 ], [ 1.25, 4.23], [ 2.1, 4.3 ] ] )
        ValidationIds = Series( [ '4a' ] )
        TrainingIds = Series( [ '1a', '2a', '3a' ] )

        self.__Splitter.validationSplit.return_value = ( TrainingIds, ValidationIds )
        self.__Vectorizer.featureizeTrain.return_value = TrainingFeatures
        self.__Vectorizer.featureizeTest.return_value = TestFeatures
        self.__PM.classifier = 'is_cancer'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        ArgumentsLabels, _ = self.__Encoder.hotEncode.call_args_list[ 0 ]

        self.assertEqual(
            list( self.__Data[ 'is_cancer' ].filter( list( TrainingIds ) ) ),
            ArgumentsLabels[ 0 ].tolist()
        )

        ArgumentsLabels, _ = self.__Encoder.hotEncode.call_args_list[ 1 ]
        self.assertEqual(
            list( self.__Data[ 'is_cancer' ].filter( list( ValidationIds ) ) ),
            ArgumentsLabels[ 0 ].tolist()
        )

        self.assertEqual(
            2,
            len( self.__Encoder.hotEncode.call_args_list )
        )

    def test_it_hot_encodes_the_labels_for_multiclass( self ):
        TrainingFeatures = Array( [ [ 0., 2. ], [ 0.1, 0.3 ], [ 1.2, 4.3], [2.3, 4.2] ] )
        TestFeatures = Array( [ [ 0.1, 2.2 ], [ 0.13, 3.3 ], [ 1.25, 4.23], [ 2.1, 4.3 ] ] )
        ValidationIds = Series( [ '4a' ] )
        TrainingIds = Series( [ '1a', '2a', '3a' ] )

        self.__Splitter.validationSplit.return_value = ( TrainingIds, ValidationIds )
        self.__Vectorizer.featureizeTrain.return_value = TrainingFeatures
        self.__Vectorizer.featureizeTest.return_value = TestFeatures
        self.__PM.classifier = 'doid'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        ArgumentsLabels, _ = self.__Encoder.hotEncode.call_args_list[ 0 ]

        self.assertEqual(
            list( self.__Data[ 'doid' ].filter( list( TrainingIds ) ) ),
            ArgumentsLabels[ 0 ].tolist()
        )

        ArgumentsLabels, _ = self.__Encoder.hotEncode.call_args_list[ 1 ]
        self.assertEqual(
            list( self.__Data[ 'doid' ].filter( list( ValidationIds ) ) ),
            ArgumentsLabels[ 0 ].tolist()
        )

        self.assertEqual(
            2,
            len( self.__Encoder.hotEncode.call_args_list )
        )

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_collects_the_input_data_for_labels(
        self,
        DataBinding: MagicMock,
    ):
        TrainingFeatures = Array( [ [ 0., 2. ], [ 0.1, 0.3 ], [ 1.2, 4.3], [2.3, 4.2] ] )
        TestFeatures = Array( [ [ 0.1, 2.2 ], [ 0.13, 3.3 ], [ 1.25, 4.23], [ 2.1, 4.3 ] ] )

        EncodedTrainingLabels = MagicMock()
        EncodedValidationLabels = MagicMock()
        Encoded = [ EncodedTrainingLabels, EncodedValidationLabels ]

        self.__Vectorizer.featureizeTrain.return_value = TrainingFeatures
        self.__Vectorizer.featureizeTest.return_value = TestFeatures
        self.__Encoder.hotEncode.side_effect = lambda _ : Encoded.pop( 0 )

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        ArgumentsLabels, _ = DataBinding.call_args_list[ 1 ]

        self.assertEqual(
            EncodedTrainingLabels,
            ArgumentsLabels[ 0 ]
        )
        self.assertEqual(
            EncodedValidationLabels,
            ArgumentsLabels[ 1 ]
        )
        self.assertEqual(
            None,
            ArgumentsLabels[ 2 ]
        )

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_builds_a_model( self, DataBinding: MagicMock ):
        Shape = ( 2, 3 )
        Payload = MagicMock( spec = InputData )
        Payload.Training = MagicMock()
        Payload.Training.shape = Shape
        Payload.Validation = MagicMock()
        Payload.Test = MagicMock()

        DataBinding.return_value = Payload

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__MLP.buildModel.assert_called_once_with( Shape, ANY )

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_builds_the_model_without_weights( self, DataBinding: MagicMock ):
        Features = MagicMock()
        Labels = MagicMock()
        Bindings = [ Features, Labels ]

        DataBinding.side_effect = lambda _, __, ___ : Bindings.pop( 0 )
        self.__Measurer.measureClassWeights.return_value = None

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__MLP.buildModel.assert_called_once_with( ANY, None )

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_builds_the_model_with_weights( self, DataBinding: MagicMock ):
        Features = MagicMock()
        Labels = MagicMock()
        Bindings = [ Features, Labels ]
        Weights = MagicMock()

        DataBinding.side_effect = lambda _, __, ___ : Bindings.pop( 0 )
        self.__Measurer.measureClassWeights.return_value = Weights

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__MLP.buildModel.assert_called_once_with( ANY, Weights )

    def test_it_saves_the_model_structure( self ):
        Expected = "structure"

        self.__MLP.buildModel.return_value = Expected

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureModel.assert_called_once_with( Expected )

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_trains_the_model( self, DataBinding: MagicMock ):
        Features = MagicMock()
        Labels = MagicMock()
        Bindings = [ Features, Labels ]
        Weights = MagicMock()

        DataBinding.side_effect = lambda _, __, ___ : Bindings.pop( 0 )
        self.__Measurer.measureClassWeights.return_value = Weights

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__MLP.train.assert_called_once_with( Features, Labels )

    def test_it_captures_the_training_time( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureTrainingTime.assert_called_once()

    def test_it_saves_the_trainings_history( self ):
        History = MagicMock()

        self.__MLP.train.return_value = History

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureTrainingHistory.assert_called_once_with( History )

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_evaluates_not_the_training( self, DataBinding: MagicMock ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__MLP.getTrainingScore.assert_not_called()

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_saves_not_the_training_evaluation( self, DataBinding: MagicMock ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.captureEvaluationScore.assert_not_called()

    @patch( 'biomed.text_mining.text_mining_controller.InputData' )
    def test_it_predicts_on_the_given_test_data( self, DataBinding: MagicMock ):
        Features = MagicMock( spec = InputData )
        Test = MagicMock()
        Features.Training = MagicMock()
        Features.Validation = MagicMock()
        Features.Test = Test

        Labels = MagicMock()
        Bindings = [ Features, Labels ]

        DataBinding.side_effect = lambda _, __, ___ : Bindings.pop( 0 )

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__MLP.predict.assert_called_once_with( Test )

    def test_it_decodes_the_predicted_data( self ):
        Predicted = MagicMock()
        self.__MLP.predict.return_value = Predicted

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Encoder.decode.assert_called_once_with( Predicted )

    def test_it_captures_the_prediction_time( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.caputrePredictingTime.assert_called_once()

    def test_it_captures_the_predictions_for_binary( self ):
        Predictions = MagicMock()

        self.__Encoder.decode.return_value = Predictions
        self.__PM.classifier = 'is_cancer'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Evaluator.capturePredictions.call_args_list[ 0 ]

        self.assertEqual(
            Arguments[ 0 ],
            Predictions,
        )

        CleanIds = list( self.__TestData[ 'pmid' ] )
        for Index in range( 0, len( CleanIds ) ):
            CleanIds[ Index ] = CleanIds[ Index ].lstrip( 'bce1f597-48e3-42e8-a140-1675e60f34f3' )

        self.assertEqual(
            list( Arguments[ 1 ] ),
            CleanIds
        )

        self.assertEqual(
            Arguments[ 2 ],
            None
        )

        self.__Evaluator.capturePredictions.assert_called_once()

    def test_it_captures_the_predictions_for_mulitclass( self ):
        Predictions = MagicMock()

        self.__Encoder.decode.return_value = Predictions
        self.__PM.classifier = 'doid'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        Arguments, _ = self.__Evaluator.capturePredictions.call_args_list[ 0 ]

        self.assertEqual(
            Arguments[ 0 ],
            Predictions,
        )

        CleanIds = list( self.__TestData[ 'pmid' ] )
        for Index in range( 0, len( CleanIds ) ):
            CleanIds[ Index ] = CleanIds[ Index ].lstrip( 'bce1f597-48e3-42e8-a140-1675e60f34f3' )

        self.assertEqual(
            list( Arguments[ 1 ] ),
            CleanIds
        )

        self.assertEqual(
            Arguments[ 2 ],
            None
        )

        self.__Evaluator.capturePredictions.assert_called_once()

    def test_it_captures_not_the_score_of_predictions_for_binary( self ):
        self.__Encoder.decode.return_value = MagicMock()
        self.__Encoder.getCategories.return_value = MagicMock()
        self.__PM.classifier = 'is_cancer'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.score.assert_not_called()

    def test_it_captures_the_score_of_predictions_for_multiclass( self ):
        self.__Encoder.decode.return_value = MagicMock()
        self.__Encoder.getCategories.return_value = MagicMock()
        self.__PM.classifier = 'doid'

        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.capturePredictions.assert_called_once()

    def test_it_finalizes_the_evaluation_for_the_fold( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        self.__Evaluator.finalize.assert_called_once()

    def test_it_prefixes_test_ids( self ):
        MyController = TextminingController.Factory.getInstance( self.__fakeLocator )
        MyController.process(
            Data = self.__Data,
            TestData = self.__TestData,
            ShortName = MagicMock(),
            Description = MagicMock()
        )

        for Id in self.__TestData[ 'pmid' ]:
            self.assertTrue( Id.startswith( self.__Prefix ) )
