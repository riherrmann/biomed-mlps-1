import unittest
from unittest.mock import MagicMock, patch
from biomed.vectorizer.selector.selector import Selector
from biomed.vectorizer.selector.selector_manager import SelectorManager
from biomed.properties_manager import PropertiesManager

class SelectorManagerSpec( unittest.TestCase ):
    def setUp( self ):
        self.__DP = patch(
            'biomed.vectorizer.selector.selector_manager.DependencySelector',
            spec = Selector
        )

        self.__FP = patch(
            'biomed.vectorizer.selector.selector_manager.FactorSelector',
            spec = Selector
        )

        self.__RP = patch(
            'biomed.vectorizer.selector.selector_manager.RegressionSelector',
            spec = Selector
        )

        self.__LRP = patch(
            'biomed.vectorizer.selector.selector_manager.LogisticRegressionSelector',
            spec = Selector
        )

        self.__D = self.__DP.start()
        self.__F = self.__FP.start()
        self.__R = self.__RP.start()
        self.__LR = self.__LRP.start()

        self.__ReferenceSelector = MagicMock( spec = Selector )
        self.__D.return_value = self.__ReferenceSelector
        self.__PM = PropertiesManager()

    def tearDown( self ):
        self.__DP.stop()
        self.__FP.stop()
        self.__RP.stop()
        self.__LRP.stop()

    def __fakeLocator( self, _, __ ):
        return self.__PM

    def test_it_is_a_selector( self ):
        MySelector = SelectorManager.Factory.getInstance( self.__fakeLocator )
        self.assertTrue( isinstance( MySelector, Selector ) )

    def test_it_depends_on_properties( self ):
        def fakeLocator( ServiceKey, Type ):
            if ServiceKey != "properties":
                raise RuntimeError( "Unexpected ServiceKey" )

            if Type != PropertiesManager:
                raise RuntimeError( "Unexpected Type" )

            return PropertiesManager()

        ServiceGetter = MagicMock()
        ServiceGetter.side_effect = fakeLocator

        SelectorManager.Factory.getInstance( ServiceGetter )
        ServiceGetter.assert_called_once()

    def test_it_uses_the_given_selectors( self ):
        Selectors = {
            "dependency": self.__D,
            "factor": self.__F,
            "regression": self.__R,
            "logisticRegression": self.__LR,
        }

        for SelectorKey in Selectors:
            PM = PropertiesManager()
            PM.selection[ 'type' ] = SelectorKey

            def fakeLocator( _, __ ):
                return PM

            ServiceGetter = MagicMock()
            ServiceGetter.side_effect = fakeLocator

            MyManager = SelectorManager.Factory.getInstance( ServiceGetter )
            MyManager.build( MagicMock(), MagicMock(), MagicMock() )

            Selectors[ SelectorKey ].assert_called_once_with( PM )

    def test_it_builds_the_given_selector( self ):
        self.__PM.selection[ 'type' ] = "dependency"

        ServiceGetter = MagicMock()
        ServiceGetter.side_effect = self.__fakeLocator

        X = MagicMock()
        Y = MagicMock()
        Weights = MagicMock()

        MyManager = SelectorManager.Factory.getInstance( ServiceGetter )
        MyManager.build( X, Y, Weights )

        self.__ReferenceSelector.build.assert_called_once_with( X, Y, Weights )

    def test_it_reflects_the_given_features_if_no_selector_was_selected( self ):
        def fakeLocator( _, __ ):
            PM = PropertiesManager()
            PM.selection[ 'type' ] = None
            return PM

        Expected = MagicMock()
        Expected.toarray.return_value = Expected

        MySelector = SelectorManager.Factory.getInstance( fakeLocator )
        MySelector.build( MagicMock(), MagicMock(), MagicMock() )
        self.assertEqual(
            Expected,
            MySelector.select( Expected )
        )

    def test_it_delegates_the_given_features_to_the_selector( self ):
        self.__PM.selection[ 'type' ] = "dependency"

        ServiceGetter = MagicMock()
        ServiceGetter.side_effect = self.__fakeLocator

        X = MagicMock()

        MySelector = SelectorManager.Factory.getInstance( ServiceGetter )
        MySelector.build( MagicMock(), MagicMock(), MagicMock() )
        MySelector.select( X )
        self.__ReferenceSelector.select.assert_called_once_with( X )

    def test_it_returns_the_selection( self ):
        self.__PM.selection[ 'type' ] = "dependency"

        ServiceGetter = MagicMock()
        ServiceGetter.side_effect = self.__fakeLocator

        Selection = MagicMock()
        self.__ReferenceSelector.select.return_value = Selection

        MySelector = SelectorManager.Factory.getInstance( ServiceGetter )
        MySelector.build( MagicMock(), MagicMock(), MagicMock() )
        self.assertEqual(
            Selection,
            MySelector.select( MagicMock() )
        )

    def test_it_reflects_the_given_features_labels_if_no_selector_was_selected( self ):
        def fakeLocator( _, __ ):
            PM = PropertiesManager()
            PM.selection[ 'type' ] = None
            return PM

        Expected = MagicMock()

        MySelector = SelectorManager.Factory.getInstance( fakeLocator )
        MySelector.build( MagicMock(), MagicMock(), MagicMock() )
        self.assertEqual(
            Expected,
            MySelector.getSupportedFeatures( Expected )
        )

    def test_it_delegates_the_given_FeaturesNames_to_the_selector( self ):
        self.__PM.selection[ 'type' ] = "dependency"

        ServiceGetter = MagicMock()
        ServiceGetter.side_effect = self.__fakeLocator

        Names = MagicMock()

        MySelector = SelectorManager.Factory.getInstance( ServiceGetter )
        MySelector.build( MagicMock(), MagicMock(), MagicMock() )
        MySelector.getSupportedFeatures( Names )
        self.__ReferenceSelector.getSupportedFeatures.assert_called_once_with( Names )

    def test_it_returns_the_selected_FeaturesNames( self ):
        self.__PM.selection[ 'type' ] = "dependency"

        ServiceGetter = MagicMock()
        ServiceGetter.side_effect = self.__fakeLocator

        SelectedFeatures = MagicMock()
        self.__ReferenceSelector.getSupportedFeatures.return_value = SelectedFeatures

        MySelector = SelectorManager.Factory.getInstance( ServiceGetter )
        MySelector.build( MagicMock(), MagicMock(), MagicMock() )
        self.assertEqual(
            SelectedFeatures,
            MySelector.getSupportedFeatures( MagicMock() )
        )
