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

    def test_it_is_convertable_to_a_dict_as_a_shallow_copy( self ):
        Manager = PropertiesManager()
        self.assertFalse( isinstance( Manager, dict ) )
        MD = Manager.toDict()
        self.assertTrue( isinstance( MD, dict ) )

        for Key in MD.keys():
            self.assertEqual(
                Manager[ Key ],
                MD[ Key ]
            )
