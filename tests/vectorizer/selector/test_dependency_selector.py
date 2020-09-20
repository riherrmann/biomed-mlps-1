from biomed.vectorizer.selector.selector import Selector
from biomed.vectorizer.selector.dependency_selector import DependencySelector
from biomed.properties_manager import PropertiesManager
from sklearn.feature_selection import SelectKBest
import unittest
from unittest.mock import MagicMock, patch

class DependencySelectorSpec( unittest.TestCase ):
    def setUp( self ):
        self.__PM = PropertiesManager()

    def test_it_is_a_Selector( self ):
        MySelect = DependencySelector( self.__PM )
        self.assertTrue( isinstance( MySelect, Selector ) )

    @patch( 'biomed.vectorizer.selector.dependency_selector.SelectKBest' )
    @patch( 'biomed.vectorizer.selector.dependency_selector.chi2' )
    def test_it_uses_chi2_and_the_given_amount_of_samples( self, chi2: MagicMock, KBest: MagicMock ):
        MySelect = DependencySelector( self.__PM )

        self.__PM.selection[ 'amountOfFeatures' ] = 42

        MySelect.build( MagicMock(), MagicMock(), MagicMock() )

        KBest.assert_called_once_with(
            chi2,
            k = self.__PM.selection[ 'amountOfFeatures' ]
        )

    def test_it_fails_to_return_the_supported_features_if_a_selector_was_not_builded( self ):
        MySelect = DependencySelector( self.__PM )
        with self.assertRaises( RuntimeError, msg = "The selector must be builded before using it" ):
            MySelect.getSupportedFeatures( MagicMock() )

    @patch( 'biomed.vectorizer.selector.dependency_selector.SelectKBest' )
    def test_it_fetches_the_selected_feature_indicies( self, KBest: MagicMock ):
        MySelect = DependencySelector( self.__PM )
        Selector = MagicMock( spec = SelectKBest )
        KBest.return_value = Selector

        MySelect.build( MagicMock(), MagicMock(), MagicMock() )
        MySelect.getSupportedFeatures( MagicMock() )

        Selector.get_support.assert_called_once_with( indices = True )


    @patch( 'biomed.vectorizer.selector.dependency_selector.SelectKBest' )
    def test_it_returns_the_filtered_FeatureNames( self, KBest: MagicMock ):
        MySelect = DependencySelector( self.__PM )
        Selector = MagicMock( spec = SelectKBest )
        FeatureNames = [ "a", "b", "c", "d", "e", "f" ]
        AcceptedKeys = [ 1, 2, 4 ]

        KBest.return_value = Selector
        Selector.get_support.return_value = AcceptedKeys

        MySelect.build( MagicMock(), MagicMock(), MagicMock() )
        self.assertListEqual(
            [ "b", "c", "e" ],
            MySelect.getSupportedFeatures( FeatureNames )
        )
