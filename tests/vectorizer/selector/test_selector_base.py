import unittest
from unittest.mock import MagicMock
from biomed.vectorizer.selector.selector_base import SelectorBase
from numpy import array as Array

class SelectorBaseSpec( unittest.TestCase ):
    class StubbedSelector( SelectorBase ):
        def __init__( self, GivenSelector: MagicMock ):
            self.__GivenSelector = GivenSelector
            super( SelectorBaseSpec.StubbedSelector, self ).__init__()

        def _assembleSelector( self ):
            self._Selector = self.__GivenSelector

    def test_it_builds_a_selector( self ):
        GivenSelectorModel = MagicMock()
        X = MagicMock()
        Y = MagicMock()

        Selector = SelectorBaseSpec.StubbedSelector( GivenSelectorModel )
        Selector.build( X, Y )

        GivenSelectorModel.fit.assert_called_once_with( X, Y )

    def test_it_fails_if_a_selector_was_not_builded( self ):
        Selector = SelectorBaseSpec.StubbedSelector( MagicMock() )
        with self.assertRaises( RuntimeError, msg = "The selector must be builded before using it" ):
            Selector.select( MagicMock() )

    def test_it_returns_the_selected_features( self ):
        Expected = Array( [ 1, 2, 3, 4, 5 ] )
        GivenSelectorModel = MagicMock()
        GivenSelectorModel.transform.return_value = Expected

        X = MagicMock()
        Y = MagicMock()

        Selector = SelectorBaseSpec.StubbedSelector( GivenSelectorModel )
        Selector.build( X, Y )
        self.assertListEqual(
            list( Expected ),
            list( Selector.select( X ) )
        )

        GivenSelectorModel.transform.assert_called_once_with( X )
