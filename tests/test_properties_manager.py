import unittest
from biomed.properties_manager import PropertiesManager

class PropertiesManagerSpec( unittest.TestCase ):
    def test_it_has_access_like_a_dict( self ):
        Manager = PropertiesManager()
        ExpectedValue = 42
        Manager.preprocessing[ "workers" ] = ExpectedValue
        self.assertEqual(
            ExpectedValue,
            Manager[ "preprocessing" ][ "workers" ]
        )

    def test_it_sets_values_like_a_dict( self ):
        Manager = PropertiesManager()
        ExpectedValue = 23
        Manager[ "preprocessing" ][ "workers" ] = ExpectedValue
        self.assertEqual(
            ExpectedValue,
            Manager.preprocessing[ "workers" ]
        )
