import unittest
from unittest.mock import patch, MagicMock
from biomed.utils.simple_file_writer import SimpleFileWriter
from biomed.utils.file_writer import FileWriter

class SimpleFileWriterSpec( unittest.TestCase ):
    def setUp( self ):
        self.__openM = patch( 'biomed.utils.simple_file_writer.open' )
        self.__open = self.__openM.start()

    def tearDown( self ):
        self.__openM.stop()

    def test_it_is_a_FileWriter( self ):
        MyWriter = SimpleFileWriter.Factory.getInstance()
        self.assertTrue( isinstance( MyWriter, FileWriter ) )

    def test_it_writes_the_given_list_as_lines( self ):
        FileName = "abc.txt"
        Content = [ "1", "2", "3" ]

        File = MagicMock()
        self.__open.return_value = File
        File.__enter__.return_value = File

        MyWriter = SimpleFileWriter.Factory.getInstance()
        MyWriter.write( FileName, Content )

        self.__open.assert_any_call( FileName, 'x' )
        File.writelines.assert_called_once_with( Content )
