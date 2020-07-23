import unittest
from unittest.mock import MagicMock, patch
from biomed.vectorizer.selector.selector import Selector
from biomed.vectorizer.selector.selector_manager import SelectorManager
from biomed.properties_manager import PropertiesManager

class SelectorManagerSpec( unittest.TestCase ):
    def __fakeLocator( self, _, __ ):
        return PropertiesManager()

    @patch( 'biomed.vectorizer.selector.selector_manager.Services.getService' )
    def test_it_is_a_selector( self, ServiceGetter: MagicMock ):
        ServiceGetter.side_effect = self.__fakeLocator

        MySelector = SelectorManager.Factory.getInstance()
        self.assertTrue( isinstance( MySelector, Selector ) )

    @patch( 'biomed.vectorizer.selector.selector_manager.Services.getService' )
    def test_it_depends_on_properties( self, ServiceGetter: MagicMock ):
        def fakeLocator( ServiceKey, Type ):
            if ServiceKey != "properties":
                raise RuntimeError( "Unexpected ServiceKey" )

            if Type != PropertiesManager:
                raise RuntimeError( "Unexpected Type" )

            return PropertiesManager()

        ServiceGetter.side_effect = fakeLocator

        SelectorManager.Factory.getInstance()
        ServiceGetter.assert_called_once()

    @patch( 'biomed.vectorizer.selector.selector_manager.Services.getService' )
    def test_it_uses_the_given_selectors( self, ServiceGetter: MagicMock ):
        pass

    @patch( 'biomed.vectorizer.selector.selector_manager.Services.getService' )
    def test_it_reflects_the_given_features_if_no_selector_was_selected( self, ServiceGetter: MagicMock ):
        def fakeLocator( _, __ ):
            PM = PropertiesManager()
            PM.selection[ 'type' ] = None
            return PM

        ServiceGetter.side_effect = fakeLocator
        Expected = MagicMock()

        MySelector = SelectorManager.Factory.getInstance()
        MySelector.build( MagicMock(), MagicMock() )
        self.assertEqual(
            Expected,
            MySelector.select( Expected )
        )
