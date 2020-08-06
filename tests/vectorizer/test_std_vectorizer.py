from biomed.vectorizer.vectorizer import Vectorizer
from biomed.vectorizer.std_vectorizer import StdVectorizer
from biomed.vectorizer.selector.selector import Selector
from biomed.properties_manager import PropertiesManager
import unittest
from sklearn.feature_extraction.text import TfidfVectorizer
from unittest.mock import MagicMock, patch
from numpy import float64

class StdVectorizerSpec( unittest.TestCase ):
    def setUp( self ):
        self.__PM = PropertiesManager()
        self.__Selector = MagicMock( spec = Selector )


    def __fakeLocator( self, ServiceKey: str, __ ):
        if ServiceKey == "properties":
            return self.__PM
        else:
            return self.__Selector

    def test_it_is_a_vectorizer( self ):
        MyVec = StdVectorizer.Factory.getInstance( self.__fakeLocator )
        self.assertTrue( isinstance( MyVec, Vectorizer ) )

    def test_it_depends_on_the_properties_manager_and_selector_manager( self ):
        def fakeLocator( ServiceKey: str, ExpectedType: object ):
            if ServiceKey != "properties" and ServiceKey != "vectorizer.selector":
                raise RuntimeError( "Unknown ServiceKey" )

            if ExpectedType != PropertiesManager and ExpectedType != Selector:
                raise RuntimeError( "Unknown ServiceType" )

        ServiceGetter = MagicMock()
        ServiceGetter.side_effect = fakeLocator

        StdVectorizer.Factory.getInstance( ServiceGetter )

        self.assertEqual(
            2,
            ServiceGetter.call_count
        )

    @patch( 'biomed.vectorizer.std_vectorizer.TfidfVectorizer' )
    def test_it_initializes_the_tfidf_vectorizer( self, Vect: MagicMock ):
        self.__PM.vectorizing = dict(
            min_df = 2,
            max_df = 0.95,
            max_features = 200000,
            ngram_range = ( 1, 4 ),
            sublinear_tf = True,
            binary = False,
            analyzer = 'word', #{‘word’, ‘char’, ‘char_wb’}
            use_idf = True,
            norm = 'l2',
            smooth_idf = True,
        )

        MyVectorizer = StdVectorizer.Factory.getInstance( self.__fakeLocator )
        MyVectorizer.featureizeTrain( MagicMock(), MagicMock() )

        Vect.assert_called_once_with(
            analyzer = self.__PM.vectorizing[ 'analyzer' ],
            min_df = self.__PM.vectorizing[ 'min_df' ],
            max_df = self.__PM.vectorizing[ 'max_df' ],
            max_features = self.__PM.vectorizing[ 'max_features' ],
            ngram_range = self.__PM.vectorizing[ 'ngram_range' ],
            use_idf = self.__PM.vectorizing[ 'use_idf' ],
            smooth_idf = self.__PM.vectorizing[ 'smooth_idf' ],
            sublinear_tf = self.__PM.vectorizing[ 'sublinear_tf' ],
            norm = self.__PM.vectorizing[ 'norm' ],
            binary = self.__PM.vectorizing[ 'binary' ],
            dtype = float64,
        )

    @patch( 'biomed.vectorizer.std_vectorizer.TfidfVectorizer' )
    def test_it_fits_and_transform_the_given_training_features( self, Vect: MagicMock ):
        Tdidf = MagicMock( spec = TfidfVectorizer )
        Vect.return_value = Tdidf

        X = MagicMock()
        Y = MagicMock()

        MyVectorizer = StdVectorizer.Factory.getInstance( self.__fakeLocator )
        MyVectorizer.featureizeTrain( X, Y )

        Tdidf.fit_transform.assert_called_once_with( X )

    @patch( 'biomed.vectorizer.std_vectorizer.TfidfVectorizer' )
    def test_it_delegates_the_extracted_training_features_and_labels_to_the_selector( self, Vect: MagicMock ):
        Tdidf = MagicMock( spec = TfidfVectorizer )
        Vect.return_value = Tdidf

        X = MagicMock()
        Y = MagicMock()
        F = MagicMock()

        Tdidf.fit_transform.return_value = F

        MyVectorizer = StdVectorizer.Factory.getInstance( self.__fakeLocator )
        MyVectorizer.featureizeTrain( X, Y )

        self.__Selector.build.assert_called_once_with( F, Y )

    @patch( 'biomed.vectorizer.std_vectorizer.TfidfVectorizer' )
    def test_it_selects_from_the_extracted_training_features( self, Vect: MagicMock ):
        Tdidf = MagicMock( spec = TfidfVectorizer )
        Vect.return_value = Tdidf

        F = MagicMock()
        Tdidf.fit_transform.return_value = F

        MyVectorizer = StdVectorizer.Factory.getInstance( self.__fakeLocator )
        MyVectorizer.featureizeTrain( MagicMock(), MagicMock() )

        self.__Selector.select.assert_called_once_with( F )

    @patch( 'biomed.vectorizer.std_vectorizer.TfidfVectorizer' )
    def test_it_returns_the_selection_of_the_extracted_training_features( self, Vect: MagicMock ):
        Selection = MagicMock()

        self.__Selector.select.return_value = Selection

        MyVectorizer = StdVectorizer.Factory.getInstance( self.__fakeLocator )

        self.assertEqual(
            Selection,
            MyVectorizer.featureizeTrain( MagicMock(), MagicMock() )
        )

    def test_it_fails_if_no_trainings_features_had_been_extracted_before( self ):
        MyVectorizer = StdVectorizer.Factory.getInstance( self.__fakeLocator )

        with self.assertRaises( RuntimeError, msg = "You must extract trainings feature, before" ):
            MyVectorizer.featureizeTest( MagicMock() )

    @patch( 'biomed.vectorizer.std_vectorizer.TfidfVectorizer' )
    def test_it_extracts_test_features( self, Vect: MagicMock ):
        Tdidf = MagicMock( spec = TfidfVectorizer )
        Vect.return_value = Tdidf

        X = MagicMock()

        MyVectorizer = StdVectorizer.Factory.getInstance( self.__fakeLocator )
        MyVectorizer.featureizeTrain( MagicMock(), MagicMock() )
        MyVectorizer.featureizeTest( X )

        Tdidf.transform.assert_called_once_with( X )

    @patch( 'biomed.vectorizer.std_vectorizer.TfidfVectorizer' )
    def test_it_selects_from_the_extracted_test_features( self, Vect: MagicMock ):
        Tdidf = MagicMock( spec = TfidfVectorizer )
        Vect.return_value = Tdidf

        F = MagicMock()
        Tdidf.transform.return_value = F

        MyVectorizer = StdVectorizer.Factory.getInstance( self.__fakeLocator )
        MyVectorizer.featureizeTrain( MagicMock(), MagicMock() )
        MyVectorizer.featureizeTest( MagicMock() )

        self.__Selector.select.assert_called_with( F )

    @patch( 'biomed.vectorizer.std_vectorizer.TfidfVectorizer' )
    def test_it_returns_the_selection_of_the_extracted_test_features( self, Vect: MagicMock ):
        Selection = MagicMock()
        self.__Selector.select.return_value = Selection

        MyVectorizer = StdVectorizer.Factory.getInstance( self.__fakeLocator )
        MyVectorizer.featureizeTrain( MagicMock(), MagicMock() )

        self.assertEqual(
            Selection,
            MyVectorizer.featureizeTest( MagicMock() )
        )

    def test_it_fails_while_returning_the_supported_features( self ):
        MyVectorizer = StdVectorizer.Factory.getInstance( self.__fakeLocator )

        with self.assertRaises( RuntimeError, msg = "You must extract trainings feature, before" ):
            MyVectorizer.getSupportedFeatures()

    @patch( 'biomed.vectorizer.std_vectorizer.TfidfVectorizer' )
    def test_it_selects_from_the_extracted_test_features_names( self, Vect: MagicMock ):
        Tdidf = MagicMock( spec = TfidfVectorizer )
        Vect.return_value = Tdidf

        F = MagicMock()
        Tdidf.get_feature_names.return_value = F

        MyVectorizer = StdVectorizer.Factory.getInstance( self.__fakeLocator )
        MyVectorizer.featureizeTrain( MagicMock(), MagicMock() )
        MyVectorizer.getSupportedFeatures()

        self.__Selector.getSupportedFeatures.assert_called_with( F )

    @patch( 'biomed.vectorizer.std_vectorizer.TfidfVectorizer' )
    def test_it_returns_the_selection_of_the_extracted_test_features_names( self, Vect: MagicMock ):
        Selection = MagicMock()
        self.__Selector.getSupportedFeatures.return_value = Selection

        MyVectorizer = StdVectorizer.Factory.getInstance( self.__fakeLocator )
        MyVectorizer.featureizeTrain( MagicMock(), MagicMock() )

        self.assertEqual(
            Selection,
            MyVectorizer.getSupportedFeatures()
        )
