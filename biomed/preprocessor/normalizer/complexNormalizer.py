from biomed.preprocessor.normalizer.normalizer import Normalizer
from biomed.preprocessor.normalizer.normalizer import NormalizerFactory
import os as OS
import re as RegEx
import subprocess as Process

class ComplexNormalizer( Normalizer ):
    __Pattern = RegEx.compile( r'\n+' )
    __PathToJar = OS.path.abspath(
        OS.path.join(
            OS.path.dirname( __file__ ), '..', '..', '..', 'nlpclient', 'client.jar' )
    )

    def apply( self, StackOfDocument: list, Flags: str ) -> list:
        return self.__run(
            self.__startSubprocess( Flags ),
            self.__glueDocumentsTogether( StackOfDocument ).encode( 'UTF-8' ),
        )

    def __splitAndClean( self, Parsed: str ) -> list:
        Documents = Parsed.split( "\n" )
        if not Documents[ -1 ]:
            Documents.pop()

        return Documents

    def __run( self, Sub: Process.Popen, Documents: str ) -> list:
        Stdout, Stderr = Sub.communicate( Documents )
        Stderr = Stderr.decode( 'UTF-8' )
        if Stderr:
            raise RuntimeError( Stderr )

        return self.__splitAndClean( Stdout.decode( 'UTF-8' ) )

    def __startSubprocess( self, Flags: str ) -> Process.Popen:
        return Process.Popen(
            [
                "java",
                "-jar",
                ComplexNormalizer.__PathToJar,
                "-f",
                Flags
            ],
            stdin = Process.PIPE,
            stdout = Process.PIPE,
            stderr = Process.PIPE,
        )

    def __adjustFormat( self, Document: str ) -> str:
        return ComplexNormalizer.__Pattern.sub( r' ', Document ).strip()

    def __glueDocumentsTogether( self, StackOfDocument: list ) -> tuple:
        StackOfDocument = list( StackOfDocument )

        for Index in range( 0, len( StackOfDocument ) ):
            StackOfDocument[ Index ] = self.__adjustFormat( StackOfDocument[ Index ] )

        return "\n".join( StackOfDocument ).strip()

    class Factory( NormalizerFactory ):
        def getApplicableFlags( self ) -> list:
            return [ "a", "d", "n", "v"  ]

        def getInstance( self ) -> Normalizer:
            return ComplexNormalizer()
