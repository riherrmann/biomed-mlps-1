from biomed.measurer.std_measurer import StdMeasurer
from biomed.measurer.measurer import Measurer
from biomed.properties_manager import PropertiesManager
import unittest
from unittest.mock import MagicMock, patch
from numpy import array as Array

class StdMeasurerSpec( unittest.TestCase ):
    def setUp( self ):
        self.__PM = PropertiesManager()


    def __fakeLocator( self, ServiceKey: str, __ ):
        return self.__PM

    def test_it_is_a_measurer( self ):
        MyMeasurer = StdMeasurer.Factory.getInstance( self.__fakeLocator )
        self.assertTrue( isinstance( MyMeasurer, Measurer ) )

    def test_it_depends_on_properties( self ):
        def fakeGetter( ServiceKey: str, ExpectedType ):
             Assigment = {
                 "properties": ( self.__PM, PropertiesManager ),
             }

             if ExpectedType != Assigment[ ServiceKey ][ 1 ]:
                 raise RuntimeError( "Unexpected Depenendcie Type" )

             return Assigment[ ServiceKey ][ 0 ]

        ServiceGetter = MagicMock()
        ServiceGetter.side_effect = fakeGetter

        StdMeasurer.Factory.getInstance( ServiceGetter )
        self.assertEqual(
            1,
            ServiceGetter.call_count
        )

    @patch( 'biomed.measurer.std_measurer.weightClasses' )
    def test_it_returns_none_if_no_class_weights_are_in_usage( self, weightFunc: MagicMock ):
        self.__PM.weights[ 'use_class_weights' ] = False
        MyMeasurer = StdMeasurer.Factory.getInstance( self.__fakeLocator )

        self.assertEqual(
            None,
            MyMeasurer.measureClassWeights( MagicMock(), MagicMock() )
        )

        self.assertFalse( weightFunc.called )


    @patch( 'biomed.measurer.std_measurer.weightClasses' )
    def test_it_delegates_the_inputs_to_a_weight_function_if_classWeights_are_active( self, weightFunc: MagicMock ):
        self.__PM.weights[ 'use_class_weights' ] = True
        Classes = MagicMock()
        Actual = MagicMock()

        MyMeasurer = StdMeasurer.Factory.getInstance( self.__fakeLocator )
        MyMeasurer.measureClassWeights( Classes, Actual )

        weightFunc.assert_called_once_with(
            'balanced',
            Classes,
            Actual,
        )

    @patch( 'biomed.measurer.std_measurer.weightClasses' )
    def test_it_maps_the_weights_to_the_given_classes( self, weightFunc: MagicMock ):
        self.__PM.weights[ 'use_class_weights' ] = True

        Classes = Array( [ 0, 1 ] )
        Weights = Array( [ 0.23, 0.42 ] )

        weightFunc.return_value = Weights
        MyMeasurer = StdMeasurer.Factory.getInstance( self.__fakeLocator )
        self.assertDictEqual(
            { 0: 0.23, 1: 0.42 },
            MyMeasurer.measureClassWeights( Classes, MagicMock() )
        )
