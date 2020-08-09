from biomed.utils.file_writer import FileWriter, FileWriterFactory
import json as JSON

class JSONFileWriter( FileWriter ):
    def write( self, FileName: str, Content ):
        with open( FileName, 'x' ) as File:
            JSON.dump( Content, File )

    class Factory( FileWriterFactory ):
        @staticmethod
        def getInstance() -> FileWriter:
            return JSONFileWriter()
