import unittest
from unittest.mock import patch, MagicMock
from pandas import Series
from biomed.splitter.std_splitter import StdSplitter
from biomed.splitter.splitter import Splitter
from biomed.properties_manager import PropertiesManager
from numpy import array as Array

class StdSplitterSpec( unittest.TestCase ):

    def setUp( self ):
        self.__PM = PropertiesManager()
        self.__PM.splitting[ 'folds' ] = 1

    def __fakeLocator( self, _, __ ):
        return self.__PM

    def test_it_is_a_splitter( self ):
        MySplitter = StdSplitter.Factory.getInstance( self.__fakeLocator )
        self.assertTrue( isinstance( MySplitter, Splitter ) )

    def test_it_depends_on_properties( self ):
        def fakeLocator( ServiceKey: str, Type ):
            if ServiceKey != 'properties':
                raise RuntimeError( 'Unknown ServiceKey' )
            if Type != PropertiesManager:
                raise RuntimeError( 'Unknown ServiceType' )

        ServiceGetter = MagicMock()
        ServiceGetter.side_effect = fakeLocator

        StdSplitter.Factory.getInstance( ServiceGetter )
        ServiceGetter.assert_called_once()

    @patch( 'biomed.splitter.std_splitter.simpleSplit' )
    def test_it_splits_test_and_trainings_data(
        self,
        split: MagicMock
    ):
        Expected = [ MagicMock(), MagicMock() ]
        X = MagicMock( spec = Series )
        Y = MagicMock( spec = Series )
        split.return_value = Expected

        MySplitter = StdSplitter.Factory.getInstance( self.__fakeLocator )
        self.assertListEqual(
            [ tuple( Expected ) ],
            MySplitter.trainingSplit( X, Y )
        )

        split.assert_called_once_with(
            X,
            test_size = self.__PM.splitting[ 'test' ],
            random_state = self.__PM.splitting[ 'seed' ],
            shuffle = True,
            stratify = Y
        )

    @patch( 'biomed.splitter.std_splitter.ComplexSplitter' )
    def test_it_makes_a_kfold_split_on_test_and_trainings_data(
        self,
        Splitter: MagicMock
    ):
        Expected =[
            ( Series( [ 'a', 'c' ] ), Series( [ 'b', 'd' ] ) ),
            ( Series( [ 'b', 'd' ] ), Series( [ 'a', 'c' ] ) )
        ]

        X = Series( [ 'a', 'b', 'c', 'd' ] )
        Y = Series( [ 1, 2, 1, 2 ] )

        Splitter.return_value = Splitter
        Splitter.split.return_value = [
            ( Array( [ 0, 2 ] ), Array( [ 1, 3 ] ) ),
            ( Array( [ 1, 3 ] ), Array( [ 0, 2 ] ) )
        ]

        self.__PM.splitting[ 'folds' ] = 2

        MySplitter = StdSplitter.Factory.getInstance( self.__fakeLocator )
        Splitted = MySplitter.trainingSplit( X, Y )
        for Index in range( 0, len( Expected ) ):
            self.assertEqual(
                list( Expected[ Index ][ 0 ] ),
                list( Splitted[ Index ][ 0 ] )
            )

            self.assertEqual(
                list( Expected[ Index ][ 1 ] ),
                list( Splitted[ Index ][ 1 ] )
            )

        Splitter.assert_called_once_with(
            n_splits = self.__PM.splitting[ 'folds' ],
            test_size = self.__PM.splitting[ 'test' ],
            random_state = self.__PM.splitting[ 'seed' ]
        )

        Splitter.split.assert_called_once_with(
            X,
            Y
        )

    @patch( 'biomed.splitter.std_splitter.simpleSplit' )
    def test_it_splits_trainings_and_validation_data(
        self,
        split: MagicMock
    ):
        Expected = [ MagicMock(), MagicMock() ]
        X = MagicMock( spec = Series )
        Y = MagicMock( spec = Series )
        split.return_value = Expected

        MySplitter = StdSplitter.Factory.getInstance( self.__fakeLocator )
        self.assertEqual(
            tuple( Expected ),
            MySplitter.validationSplit( X, Y )
        )

        split.assert_called_once_with(
            X,
            test_size = self.__PM.splitting[ 'validation' ],
            random_state = self.__PM.splitting[ 'seed' ],
            shuffle = True,
            stratify = Y
        )
