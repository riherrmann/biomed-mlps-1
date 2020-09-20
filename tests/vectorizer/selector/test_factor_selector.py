from biomed.vectorizer.selector.selector import Selector
from biomed.vectorizer.selector.factor_selector import FactorSelector
from biomed.properties_manager import PropertiesManager
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import unittest
from unittest.mock import MagicMock, patch, ANY

class FactorSelectorSpec( unittest.TestCase ):
    def setUp( self ):
        self.__PM = PropertiesManager()

    def test_it_is_a_Selector( self ):
        MySelect = FactorSelector( self.__PM )
        self.assertTrue( isinstance( MySelect, Selector ) )

    @patch( 'biomed.vectorizer.selector.factor_selector.SelectFromModel' )
    @patch( 'biomed.vectorizer.selector.factor_selector.ExtraTreesClassifier' )
    def test_it_uses_the_given_amount_of_samples( self, TreeModel: MagicMock, ModelSelector: MagicMock ):
        MySelect = FactorSelector( self.__PM )

        self.__PM.selection[ 'amountOfFeatures' ] = 42

        MySelect.build( MagicMock(), MagicMock(), None )

        TreeModel.assert_called_once_with(
            n_estimators = ANY,
            class_weight = ANY,
            max_features = self.__PM.selection[ 'amountOfFeatures' ],
        )

    @patch( 'biomed.vectorizer.selector.factor_selector.SelectFromModel' )
    @patch( 'biomed.vectorizer.selector.factor_selector.ExtraTreesClassifier' )
    def test_it_uses_the_given_amount_of_trees( self, TreeModel: MagicMock, ModelSelector: MagicMock ):
        MySelect = FactorSelector( self.__PM )

        self.__PM.selection[ 'treeEstimators' ] = 42

        MySelect.build( MagicMock(), MagicMock(), None )

        TreeModel.assert_called_once_with(
            n_estimators = self.__PM.selection[ 'treeEstimators' ],
            class_weight = ANY,
            max_features = ANY,
        )

    @patch( 'biomed.vectorizer.selector.factor_selector.SelectFromModel' )
    @patch( 'biomed.vectorizer.selector.factor_selector.ExtraTreesClassifier' )
    def test_it_uses_the_given_class_weights( self, TreeModel: MagicMock, ModelSelector: MagicMock ):
        Weights = MagicMock()

        MySelect = FactorSelector( self.__PM )

        MySelect.build( MagicMock(), MagicMock(), Weights )

        TreeModel.assert_called_once_with(
            n_estimators = ANY,
            class_weight = Weights,
            max_features = ANY,
        )

    @patch( 'biomed.vectorizer.selector.factor_selector.SelectFromModel' )
    @patch( 'biomed.vectorizer.selector.factor_selector.ExtraTreesClassifier' )
    def test_it_builds_the_selector_from_the_tree_classifier( self, TreeModel: MagicMock, ModelSelector: MagicMock ):
        MySelect = FactorSelector( self.__PM )

        Model = MagicMock( spec = ExtraTreesClassifier )
        TreeModel.return_value = Model

        MySelect.build( MagicMock(), MagicMock(), MagicMock() )

        ModelSelector.assert_called_once_with( Model, prefit = False )

    def test_it_fails_to_return_the_supported_features_if_a_selector_was_not_builded( self ):
        MySelect = FactorSelector( self.__PM )
        with self.assertRaises( RuntimeError, msg = "The selector must be builded before using it" ):
            MySelect.getSupportedFeatures( MagicMock() )

    @patch( 'biomed.vectorizer.selector.factor_selector.SelectFromModel' )
    @patch( 'biomed.vectorizer.selector.factor_selector.ExtraTreesClassifier' )
    def test_it_fetches_the_selected_feature_indicies( self, TreeModel: MagicMock, ModelSelector: MagicMock ):
        MySelect = FactorSelector( self.__PM )
        Selector = MagicMock( spec = SelectFromModel )
        ModelSelector.return_value = Selector

        MySelect.build( MagicMock(), MagicMock(), MagicMock() )
        MySelect.getSupportedFeatures( MagicMock() )

        Selector.get_support.assert_called_once_with( indices = True )


    @patch( 'biomed.vectorizer.selector.factor_selector.SelectFromModel' )
    @patch( 'biomed.vectorizer.selector.factor_selector.ExtraTreesClassifier' )
    def test_it_returns_the_filtered_FeatureNames( self, TreeModel: MagicMock, ModelSelector: MagicMock ):
        MySelect = FactorSelector( self.__PM )
        Selector = MagicMock( spec = SelectFromModel )
        FeatureNames = [ "a", "b", "c", "d", "e", "f" ]
        AcceptedKeys = [ 1, 2, 4 ]

        ModelSelector.return_value = Selector
        Selector.get_support.return_value = AcceptedKeys

        MySelect.build( MagicMock(), MagicMock(), MagicMock() )
        self.assertListEqual(
            [ "b", "c", "e" ],
            MySelect.getSupportedFeatures( FeatureNames )
        )
