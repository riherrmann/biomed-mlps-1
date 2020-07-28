from biomed.utils.file_writer import FileWriter, FileWriterFactory
import csv as CSV

class CSVFileWriter( FileWriter ):
    def __getHeader( self, Content: dict ) -> list:
        return list( Content.keys() )

    def __writeSimpleDict( self, FileName: str, Header: list, Content: dict ):
        with open( FileName, 'x' ) as File:
            Writer = CSV.DictWriter( File, delimiter = ',', fieldnames = Header )
            Writer.writeheader()
            Writer.writerow( Content )

    def __checkDimensions( self, Content: dict ):
        Values = list( Content.values() )

        if not isinstance( Values[ 0 ], list ):
            raise RuntimeError( "The given dict has ambiguous dimentions" )

        BaseLine = len( Values[ 0 ] )
        Values.pop( 0 )

        for Value in Values:
            if not isinstance( Value, list ) or BaseLine != len( Value ):
                raise RuntimeError( "The given dict has ambiguous dimentions" )

    def __mapEntry( self, Index: int, Keys: list, Content: dict ) -> dict:
        Entry = {}

        for Key in Keys:
            Entry[ Key ] = Content[ Key ][ Index ]

        return Entry

    def __remap( self, Header: list, Content: dict ) -> list:
        self.__checkDimensions( Content )
        Remaped = []
        Iterations = len( Content[ Header[ 0 ] ] )

        for Index in range( 0, Iterations ):
            Remaped.append( self.__mapEntry( Index, Header, Content ) )

        return Remaped

    def __write2DDict( self, FileName: str, Header: list, Content: dict ):
        Remaped = self.__remap( Header, Content )
        with open( FileName, 'x' ) as File:
            Writer = CSV.DictWriter( File, delimiter = ',', fieldnames = Header )
            Writer.writeheader()
            Writer.writerows( Remaped )

    def __is2D( self, Content: dict ):
        for Value in Content.values():
            if isinstance( Value, list ):
                return True
        else:
            return False

    def __checkNestedDimension( self, Content: dict ):
        Values = list( Content.values() )

        if not isinstance( Values[ 0 ], dict ):
            raise RuntimeError( "The given dict has ambiguous dimentions" )

        BaseLine = list( Values[ 0 ].keys() )
        Values.pop( 0 )

        for Value in Values:
            if not isinstance( Value, dict ) or BaseLine != list( Value.keys() ):
                raise RuntimeError( "The given dict has ambiguous dimentions" )

    def __getNestedColumnNames( self, Content: dict ) -> list:
        return [ str( I ) for I in list( range( len( list( Content.values() )[ 0 ].keys() )+1 ) ) ]

    def __getNestedHeader( self, Content: dict ) -> dict:
        Header = { '0': '' }
        MapKey = 1
        ContentKeys = list( Content.values() )[ 0 ].keys()

        for Key in ContentKeys:
            Header[ str( MapKey ) ] = Key
            MapKey += 1

        return Header


    def __mapNestedEntry( self, Key, SubKeys: list, Content: dict ) -> dict:
        Entry = { '0': Key }
        MapKey = 1

        for SubKey in SubKeys:
            Entry[ str( MapKey ) ] = Content[ Key ][ SubKey ]
            MapKey += 1

        return Entry

    def __remapNested( self, Header: list, Content: dict ) -> list:
        Remaped = []
        SubKeys =  list( Content.values() )[ 0 ].keys()

        Remaped.append( self.__getNestedHeader( Content ) )

        for Key in Content.keys():
            Remaped.append( self.__mapNestedEntry( Key, SubKeys, Content ) )

        return Remaped

    def __write2DNestedDict( self, FileName, Content: dict ):
        self.__checkNestedDimension( Content )
        ColumnNames = self.__getNestedColumnNames( Content )
        Remaped = self.__remapNested( ColumnNames, Content )

        with open( FileName, 'x' ) as File:
            Writer = CSV.DictWriter( File, delimiter = ',', fieldnames = ColumnNames )
            Writer.writerows( Remaped )

    def __isNested( self, Content: dict ):
        for Value in Content.values():
            if isinstance( Value, dict ):
                return True
        else:
            return False

    def write( self, FileName: str, Content: dict ):
        Header = self.__getHeader( Content )
        if self.__is2D( Content ):
            self.__write2DDict( FileName, Header, Content )
        elif self.__isNested( Content ):
            self.__write2DNestedDict( FileName, Content )
        else:
            self.__writeSimpleDict( FileName, Header, Content )

    class Factory( FileWriterFactory ):
        @staticmethod
        def getInstance() -> FileWriter:
            return CSVFileWriter()
