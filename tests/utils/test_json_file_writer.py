import unittest
from unittest.mock import patch, MagicMock
from biomed.utils.json_file_writer import JSONFileWriter
from biomed.utils.file_writer import FileWriter
import json as JSON

class JSONFileWriterSpec( unittest.TestCase ):
    def setUp( self ):
        self.__openM = patch( 'biomed.utils.json_file_writer.open' )
        self.__open = self.__openM.start()
        self.__JSONM = patch( 'biomed.utils.json_file_writer.JSON', spec = JSON )
        self.__JSON = self.__JSONM.start()


    def tearDown( self ):
        self.__openM.stop()
        self.__JSONM.stop()

    def test_it_is_a_FileWriter( self ):
        MyWriter = JSONFileWriter.Factory.getInstance()
        self.assertTrue( isinstance( MyWriter, FileWriter ) )

    def test_it_writes_the_given_list_as_lines( self ):
        FileName = "abc.txt"
        Content = [ "1", "2", "3" ]

        File = MagicMock()
        self.__open.return_value = File
        File.__enter__.return_value = File

        MyWriter = JSONFileWriter.Factory.getInstance()
        MyWriter.write( FileName, Content )

        self.__open.assert_any_call( FileName, 'x' )
        self.__JSON.dump.assert_called_once_with(
            Content,
            File
        )
