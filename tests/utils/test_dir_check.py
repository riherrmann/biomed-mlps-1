import unittest
from unittest.mock import patch, MagicMock
from biomed.utils.dir_checker import checkDir, toAbsPath
import os as OS

class DirCheckSpec( unittest.TestCase ):
    @patch( 'biomed.utils.dir_checker.OS.path.isdir' )
    def test_it_fails_if_the_given_path_is_not_a_dir( self, isDir: MagicMock ):
        isDir.return_value = False
        Dir = "abc"
        with self.assertRaises( RuntimeError, msg = "{} not found.".format( Dir ) ):
            checkDir( Dir )

    @patch( 'biomed.utils.dir_checker.OS.path.isdir' )
    @patch( 'biomed.utils.dir_checker.OS.access' )
    def test_it_fails_if_the_dir_should_be_readable( self, access: MagicMock, isDir: MagicMock ):
        def accessLock( Path, Right ):
            if Right == OS.R_OK:
                return False
            else:
                return True

        isDir.return_value = True
        access.side_effect = accessLock
        Dir = "abc"

        with self.assertRaises( RuntimeError, msg = "{} is not readable".format( Dir ) ):
            checkDir( Dir )

    @patch( 'biomed.utils.dir_checker.OS.path.isdir' )
    @patch( 'biomed.utils.dir_checker.OS.access' )
    def test_it_fails_if_the_dir_should_be_writeable( self, access: MagicMock, isDir: MagicMock ):
        def accessLock( Path, Right ):
            if Right == OS.W_OK:
                return False
            else:
                return True

        isDir.return_value = True
        access.side_effect = accessLock
        Dir = "abc"

        with self.assertRaises( RuntimeError, msg = "{} is not writeable".format( Dir ) ):
            checkDir( Dir )

    @patch( 'biomed.utils.dir_checker.OS.path.isdir' )
    @patch( 'biomed.utils.dir_checker.OS.access' )
    def test_it_passes_if_everything_is_ok( self, access: MagicMock, isDir: MagicMock ):
        isDir.return_value = True
        access.return_value = True
        Dir = "abc"

        checkDir( Dir )

        isDir.assert_called_once_with( Dir )
        self.assertEqual(
            2,
            access.call_count
        )

    def test_it_returns_the_absoulte_path( self ):
        self.assertEqual(
            OS.path.dirname( OS.path.abspath( __file__) ),
            toAbsPath( OS.path.dirname( __file__ ) )
        )
