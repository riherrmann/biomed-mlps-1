import unittest
from unittest.mock import MagicMock, patch
from biomed.vectorizer.selector.selector import Selector
from biomed.vectorizer.selector.selector_manager import SelectorManager
from biomed.properties_manager import PropertiesManager

class SelectorManagerSpec( unittest.TestCase ):
    def __fakeLocator( self, _, __ ):
        return PropertiesManager()

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
        pass

    def test_it_reflects_the_given_features_if_no_selector_was_selected( self ):
        def fakeLocator( _, __ ):
            PM = PropertiesManager()
            PM.selection[ 'type' ] = None
            return PM

        Expected = MagicMock()
        Expected.toarray.return_value = Expected

        MySelector = SelectorManager.Factory.getInstance( fakeLocator )
        MySelector.build( MagicMock(), MagicMock() )
        self.assertEqual(
            Expected,
            MySelector.select( Expected )
        )

    def test_it_reflects_the_given_features_labels_if_no_selector_was_selected( self ):
        def fakeLocator( _, __ ):
            PM = PropertiesManager()
            PM.selection[ 'type' ] = None
            return PM

        Expected = MagicMock()

        MySelector = SelectorManager.Factory.getInstance( fakeLocator )
        MySelector.build( MagicMock(), MagicMock() )
        self.assertEqual(
            Expected,
            MySelector.getSupportedFeatures( Expected )
        )
