import unittest
from unittest.mock import MagicMock, patch
from biomed.utils.service_locator import ServiceLocator

class ServiceLocatorSpec( unittest.TestCase ):
    @patch( 'biomed.utils.service_locator.dict' )
    def test_it_sets_services( self, Internal: MagicMock ):
        Spy = {}
        Internal.return_value = Spy
        Locator = ServiceLocator()
        Locator.set( "preprocessor", "abc" )

        self.assertDictEqual(
            Spy,
            { "preprocessor": "abc" }
        )

    def test_it_fails_if_a_single_dependency_is_not_statisfied( self ):
        Locator = ServiceLocator()

        with self.assertRaises( RuntimeError, msg = "Missing dependency preprocessor" ):
            Locator.set( "miner", "abc", "preprocessor" )

    @patch( 'biomed.utils.service_locator.dict' )
    def  test_it_fails_if_a_dependency_is_not_statisfied( self, Internal: MagicMock ):
        Spy = { "dep1": "abc" }
        Internal.return_value = Spy
        Locator = ServiceLocator()

        with self.assertRaises( RuntimeError, msg = "Missing dependency preprocessor" ):
            Locator.set( "miner", "abc", [ "dep1", "preprocessor" ] )


    @patch( 'biomed.utils.service_locator.dict' )
    def test_it_fails_on_unknown_service( self, Internal: MagicMock ):
         Spy = {}
         Internal.return_value = Spy
         Locator = ServiceLocator()
         with self.assertRaises( RuntimeError, msg = "Unknown service preprocessor" ):
             Locator.get( 'preprocessor', int )

    @patch( 'biomed.utils.service_locator.dict' )
    def test_it_fails_on_unexpected_inheretance( self, Internal: MagicMock ):
        Spy = { "preprocessor": "abc" }
        Internal.return_value = Spy
        Locator = ServiceLocator()

        with self.assertRaises( RuntimeError, msg = "Broken dependency at preprocessor" ):
            Locator.get( "preprocessor", int )

    @patch( 'biomed.utils.service_locator.dict' )
    def test_it_returns_a_given_value( self, Internal: MagicMock ):
        Spy = { "preprocessor": "abc" }
        Internal.return_value = Spy
        Locator = ServiceLocator()

        self.assertEqual(
            Spy[ "preprocessor" ],
            Locator.get( "preprocessor", str )
        )
