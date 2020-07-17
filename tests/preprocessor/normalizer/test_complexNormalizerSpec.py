import os as OS
import sys as Sys

AdditionalPath = OS.path.abspath( OS.path.join( OS.path.dirname( __file__ ), '..', '..', '..' ) )
if AdditionalPath not in Sys.path:
    Sys.path.append( AdditionalPath )

import unittest
from unittest.mock import MagicMock, patch, ANY
import subprocess
from biomed.preprocessor.normalizer.complexNormalizer import ComplexNormalizer
from biomed.preprocessor.normalizer.normalizer import Normalizer

class ComplexNormalizerSpec( unittest.TestCase ):
    __Documents = [
        "A",
        "B",
        "C"
    ]

    def test_it_is_a_normalizer( self ):
        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertTrue( isinstance( MyNormal, Normalizer ) )

    @patch( 'biomed.preprocessor.normalizer.complexNormalizer.Process' )
    def test_it_initilaizes_a_new_subprocess_with_the_given_configuration( self, Sub: MagicMock ):
        Flags = "na"
        NP = MagicMock( spec = subprocess.Popen )
        Sub.Popen = NP
        Impl = MagicMock( spec = subprocess.Popen )
        NP.return_value = Impl
        Impl.communicate.return_value = ( "".encode( 'UTF-8' ), "".encode( 'UTF-8' ) )

        MyNormal = ComplexNormalizer.Factory.getInstance()
        MyNormal.apply(
            ComplexNormalizerSpec.__Documents,
            Flags
        )

        NP.assert_called_once_with(
            [
                "java",
                "-jar",
                OS.path.abspath(
                    OS.path.join(
                        OS.path.dirname( __file__ ), '..', '..', '..', 'nlpclient', 'client.jar' )
                ),
                "-f",
                Flags
            ],
            stdin = ANY,
            stdout = ANY,
            stderr = ANY,
        )

    @patch( 'biomed.preprocessor.normalizer.complexNormalizer.Process' )
    def test_it_calls_the_client_with_the_normalized_and_batched_documents( self, Sub: MagicMock ):
        NP = MagicMock( spec = subprocess.Popen )
        Impl = MagicMock( spec = subprocess.Popen )
        NP.return_value = Impl
        Sub.Popen = NP

        Impl.communicate.return_value = ( "".encode( 'UTF-8' ), "".encode( 'UTF-8' ) )

        Documents = [
            "I am \n text1",
            "I am text2",
            "I\nam\ntext3\n"
        ]

        MyNormal = ComplexNormalizer.Factory.getInstance()
        MyNormal.apply(
            Documents,
            "na"
        )

        Impl.communicate.assert_called_once_with(
            "I am   text1\nI am text2\nI am text3".encode( 'UTF-8' )
        )


    @patch( 'biomed.preprocessor.normalizer.complexNormalizer.Process' )
    def test_it_fails_if_a_error_occurs_in_the_sub_process( self, Sub: MagicMock ):
        ErrorMsg = "errror"
        NP = MagicMock( spec = subprocess.Popen )
        Impl = MagicMock( spec = subprocess.Popen )

        NP.return_value = Impl
        Sub.Popen = NP

        Impl.communicate.return_value = ( "".encode( 'UTF-8' ), ErrorMsg.encode( 'UTF-8' ) )

        MyNormal = ComplexNormalizer.Factory.getInstance()
        with self.assertRaises( RuntimeError, msg = ErrorMsg):
            MyNormal.apply(
                self.__Documents,
                "na"
            )

    @patch( 'biomed.preprocessor.normalizer.complexNormalizer.Process' )
    def test_it_returns_a_list_with_the_parsed_documents( self, Sub: MagicMock ):
        ParsedDocumentes = "text1\ntext2\ntext3"
        NP = MagicMock( spec = subprocess.Popen )
        Impl = MagicMock( spec = subprocess.Popen )

        NP.return_value = Impl
        Sub.Popen = NP

        Impl.communicate.return_value = ( ParsedDocumentes.encode( 'UTF-8' ), "".encode( 'UTF-8' ) )

        MyNormal = ComplexNormalizer.Factory.getInstance()
        self.assertListEqual(
            MyNormal.apply(
                self.__Documents,
                "na"
            ),
            [ "text1", "text2", "text3" ]
        )
