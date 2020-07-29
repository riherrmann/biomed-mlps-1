import unittest
from unittest.mock import patch, MagicMock
from biomed.utils.csv_file_writer import CSVFileWriter
from biomed.utils.file_writer import FileWriter
import csv

class CSVFileWriterSpec( unittest.TestCase ):
    def setUp( self ):
        self.__openM = patch( 'biomed.utils.csv_file_writer.open' )
        self.__open = self.__openM.start()
        self.__CSVM = patch( 'biomed.utils.csv_file_writer.CSV' )
        self.__CSV = self.__CSVM.start()

    def tearDown( self ):
        self.__openM.stop()
        self.__CSVM.stop()

    def test_it_is_a_FileWriter( self ):
        MyWriter = CSVFileWriter.Factory.getInstance()
        self.assertTrue( isinstance( MyWriter, FileWriter ) )

    def test_it_writes_simple_dicts( self ):
        FileName = "abc.txt"
        Content = { 'a': 123 }

        File = MagicMock()
        self.__open.return_value = File
        File.__enter__.return_value = File

        CSVWriter = MagicMock( spec = csv.DictWriter )
        self.__CSV.DictWriter.return_value = CSVWriter

        MyWriter = CSVFileWriter.Factory.getInstance()
        MyWriter.write( FileName, Content )

        self.__open.assert_any_call( FileName, 'x' )
        self.__CSV.DictWriter.assert_called_once_with(
            File,
            delimiter = ',',
            fieldnames = list( Content.keys() )
        )

        CSVWriter.writeheader.assert_called_once()
        CSVWriter.writerow.assert_called_once_with( Content )

    def test_it_fails_if_a_dict_has_not_the_same_dimension( self ):
        FileName = "abc.txt"
        Content = { 'a': [ 1, 2 ], 'b': 1, 'c': 2 }

        MyWriter = CSVFileWriter.Factory.getInstance()
        with self.assertRaises( RuntimeError, msg = "The given dict has ambiguous dimentions" ):
            MyWriter.write( FileName, Content )

        Content = { 'a': [ 1, 2 ], 'b': [ 1, 2 ], 'c': 2 }

        MyWriter = CSVFileWriter.Factory.getInstance()
        with self.assertRaises( RuntimeError, msg = "The given dict has ambiguous dimentions" ):
            MyWriter.write( FileName, Content )

        Content = { 'a': 1, 'b': [ 1, 2 ], 'c': [ 1, 2 ] }

        MyWriter = CSVFileWriter.Factory.getInstance()
        with self.assertRaises( RuntimeError, msg = "The given dict has ambiguous dimentions" ):
            MyWriter.write( FileName, Content )

        Content = { 'a': [ 1, 2 ], 'b': [ 1, 2 ], 'c': [ 1, 2, 3 ] }
        MyWriter = CSVFileWriter.Factory.getInstance()
        with self.assertRaises( RuntimeError, msg = "The given dict has ambiguous dimentions" ):
            MyWriter.write( FileName, Content )


        Content = { 'a': [ 1, 2, 3 ], 'b': [ 1, 2 ], 'c': [ 1, 2, 3 ] }
        MyWriter = CSVFileWriter.Factory.getInstance()
        with self.assertRaises( RuntimeError, msg = "The given dict has ambiguous dimentions" ):
            MyWriter.write( FileName, Content )

    def test_it_writes_the_remaped_dict_to_a_file( self ):
        FileName = "abc.txt"
        Content = { 'a': [ 1, 2 ], 'b': [ 3, 4 ] }

        File = MagicMock()
        self.__open.return_value = File
        File.__enter__.return_value = File

        CSVWriter = MagicMock( spec = csv.DictWriter )
        self.__CSV.DictWriter.return_value = CSVWriter

        MyWriter = CSVFileWriter.Factory.getInstance()
        MyWriter.write( FileName, Content )

        self.__open.assert_any_call( FileName, 'x' )
        self.__CSV.DictWriter.assert_called_once_with(
            File,
            delimiter = ',',
            fieldnames = list( Content.keys() )
        )

        CSVWriter.writeheader.assert_called_once()
        CSVWriter.writerows.assert_called_once_with( [
            { 'a': 1, 'b': 3 },
            { 'a': 2, 'b': 4 }
        ] )

    def test_it_fails_if_a_dict_of_dicts_has_not_the_same_dimension( self ):
        FileName = "abc.txt"
        Content = {
            '1': { 'a': 1, 'b': 3 },
            '2': { 'a': 2,  }
        }

        MyWriter = CSVFileWriter.Factory.getInstance()
        with self.assertRaises( RuntimeError, msg = "The given dict has ambiguous dimentions" ):
            MyWriter.write( FileName, Content )

        Content = {
            '1': { 'b': 3 },
            '2': { 'a': 2, 'b': 4 }
        }

        MyWriter = CSVFileWriter.Factory.getInstance()
        with self.assertRaises( RuntimeError, msg = "The given dict has ambiguous dimentions" ):
            MyWriter.write( FileName, Content )

    def test_it_writes_the_remaped_dict_of_dicts_to_a_file( self ):
        FileName = "abc.txt"
        Content = {
            '1': { 'a': 1, 'b': 3 },
            '2': { 'a': 2, 'b': 4 }
        }

        File = MagicMock()
        self.__open.return_value = File
        File.__enter__.return_value = File

        CSVWriter = MagicMock( spec = csv.DictWriter )
        self.__CSV.DictWriter.return_value = CSVWriter

        MyWriter = CSVFileWriter.Factory.getInstance()
        MyWriter.write( FileName, Content )

        self.__open.assert_any_call( FileName, 'x' )
        self.__CSV.DictWriter.assert_called_once_with(
            File,
            delimiter = ',',
            fieldnames = [ '0', '1', '2' ]
        )

        CSVWriter.writeheader.assert_not_called()
        CSVWriter.writerows.assert_called_once_with( [
            { '0': '', '1': 'a', '2': 'b' },
            { '0': '1', '1': 1, '2': 3 },
            { '0': '2', '1': 2, '2': 4 }
        ] )
