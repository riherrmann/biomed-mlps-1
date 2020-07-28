from biomed.utils.file_writer import FileWriter, FileWriterFactory

class SimpleFileWriter( FileWriter ):
    def write( self, FileName: str, Content: list ):
        with open( FileName, 'x' ) as File:
            File.writelines( Content )

    class Factory( FileWriterFactory ):
        @staticmethod
        def getInstance() -> FileWriter:
            return SimpleFileWriter()
