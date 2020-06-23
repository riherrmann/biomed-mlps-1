import unittest
from biomed.properties_manager import PropertiesManager

class PropertiesManagerSpec( unittest.TestCase ):
    def test_it_has_access_like_a_dict( self ):
        Manager = PropertiesManager()
        Manager.workers = 42
        self.assertEqual(
            Manager.workers,
            Manager[ "workers" ]
        )

    def test_it_fails_on_unknown_properties( self ):
        Manager = PropertiesManager()
        with self.assertRaises( KeyError ):
            Manager[ "neverGonnaWork" ]

    def test_it_sets_values_like_a_dict( self ):
        Manager = PropertiesManager()
        ExpectedValue = 23
        Manager[ "workers" ] = ExpectedValue
        self.assertEqual(
            ExpectedValue,
            Manager.workers
        )

    def test_it_fails_on_unknown_fiel_while_setting( self ):
        Manager = PropertiesManager()
        with self.assertRaises( KeyError ):
            Manager[ "neverGonnaWork" ] = 23
