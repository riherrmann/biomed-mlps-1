import unittest
from unittest.mock import MagicMock, patch
from keras.models import Sequential
from biomed.mlp.complex import ComplexFFN

class ComplexFFNSpec( unittest.TestCase ):
    @patch( 'biomed.mlp.complex.Sequential' )
    def test_it_compiles_the_model( self, MC: MagicMock ):
        Model = MagicMock( spec = Sequential )
        MC.return_value = Model

        Simple = ComplexFFN( MagicMock() )
        Simple.buildModel( MagicMock() )

        Model.compile.assert_called_once()

    @patch( 'biomed.mlp.complex.Sequential' )
    def test_it_returns_the_model_summary( self, MC: MagicMock ):
        Summary = "summary"
        def summarize( print_fn ):
            print_fn( Summary )

        Model = MagicMock( spec = Sequential )
        MC.return_value = Model

        Model.summary.side_effect = summarize

        Simple = ComplexFFN( MagicMock() )
        self.assertEqual(
            Simple.buildModel( MagicMock() ),
            Summary
        )
