import unittest
from unittest.mock import MagicMock
from biomed.vectorizer.selector.selector_base import SelectorBase

class SelectorBaseSpec( unittest.TestCase ):
    class StubbedSelector( SelectorBase ):
        def __init__(
            self,
            SampleSize: int,
            GivenSelector: MagicMock,
            GivenSelectedKeys: list = None,
        ):
            self.__GivenSelector = GivenSelector
            self.__GivenSelectedKeys = GivenSelectedKeys
            super( SelectorBaseSpec.StubbedSelector, self ).__init__( SampleSize )

        def _assembleSelector( self ):
            self._Selector = self.__GivenSelector

        def getSupportedFeatures( self, FeatureNames: list ) -> list:
            return self._filterFeatureNamesByIndex( FeatureNames, self.__GivenSelectedKeys )

    def test_it_builds_a_selector( self ):
        GivenSelectorModel = MagicMock()
        X = MagicMock()
        Y = MagicMock()

        Selector = SelectorBaseSpec.StubbedSelector( 2, GivenSelectorModel )
        Selector.build( X, Y )

        GivenSelectorModel.fit.assert_called_once_with( X, Y )

    def test_it_fails_if_a_selector_was_not_builded( self ):
        Selector = SelectorBaseSpec.StubbedSelector( 2, MagicMock() )
        with self.assertRaises( RuntimeError, msg = "The selector must be builded before using it" ):
            Selector.select( MagicMock() )

    def test_it_returns_the_selected_features( self ):
        Expected = MagicMock()
        GivenSelectorModel = MagicMock()
        GivenSelectorModel.transform.return_value = Expected

        X = MagicMock()
        Y = MagicMock()

        Selector = SelectorBaseSpec.StubbedSelector( 2, GivenSelectorModel )
        Selector.build( X, Y )
        self.assertListEqual(
            list( Expected ),
            list( Selector.select( X ) )
        )

        Expected.toarray.assert_called_once()
        GivenSelectorModel.transform.assert_called_once_with( X )

    def test_it_filters_given_FeaturesNames_by_index( self ):
        FeatureNames = [ 'a', 'b', 'c', 'd', 'e' ]
        Keys = [ 1, 3 ]

        Selector = SelectorBaseSpec.StubbedSelector( 2, MagicMock(), Keys )
        Selector.build( MagicMock(), MagicMock() )

        self.assertListEqual(
            [ 'b', 'd' ],
            Selector.getSupportedFeatures( FeatureNames )
        )
