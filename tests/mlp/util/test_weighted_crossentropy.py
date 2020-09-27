import unittest
from unittest.mock import MagicMock, patch
from keras.losses import Loss
from biomed.mlp.util.weighted_crossentropy import WeightedCrossentropy
from numpy import array as Array
import tensorflow as TF

class WeightedBinaryCrossentropySpec( unittest.TestCase ):
    def setUp( self ):
        self.__AdapterP = patch( 'biomed.mlp.util.weighted_crossentropy.KerasAdapter' )
        self.__Adapter = self.__AdapterP.start()
        self.__MultiplicationP = patch( 'biomed.mlp.util.weighted_crossentropy.multiply' )
        self.__Multiplication = self.__MultiplicationP.start()
        self.__NPP = patch( 'biomed.mlp.util.weighted_crossentropy.Numpy' )
        self.__NP = self.__NPP.start()
        self.__NP.ones.return_value = Array( [ [0,0],[0,0] ] )
        self.__BinP = patch( 'biomed.mlp.util.weighted_crossentropy.Binary' )
        self.__Bin = self.__BinP.start()
        self.__CatP = patch( 'biomed.mlp.util.weighted_crossentropy.Categorical' )
        self.__Cat = self.__CatP.start()
        self.__SparseP = patch( 'biomed.mlp.util.weighted_crossentropy.Sparse' )
        self.__Spare = self.__SparseP.start()

    def tearDown( self ):
        self.__Adapter.stop()
        self.__MultiplicationP.stop()
        self.__NPP.stop()
        self.__BinP.stop()
        self.__CatP.stop()
        self.__SparseP.stop()

    def test_it_is_a_instance_of_losses( self ):
        MyEntropy = WeightedCrossentropy(
            'testentropy',
            MagicMock(),
            'bin',
        )

        self.assertTrue( isinstance( MyEntropy, Loss ) )

    def test_it_initializes_a_loss_function( self  ):
        Entropies = {
            "bin": self.__Bin,
            "categorical": self.__Cat,
            "sparse": self.__Spare,
        }

        for EntropyKey in Entropies:
            WeightedCrossentropy(
                'testentropy',
                MagicMock(),
                EntropyKey,
                { 'name': 'bin_weighted' }
            )

            Entropies[ EntropyKey ].assert_called_once_with(
                name = 'bin_weighted'
            )

    def test_it_raises_an_error_if_the_labels_have_a_rank_greater_then_2( self ):
        Labels = TF.constant( [
            [
                [1, 1, 1],
                [2, 2, 2]
            ],
            [
                [3, 3, 3],
                [4, 4, 4]
            ]
        ] )

        Predictions = TF.constant( [
            [
                [1, 1, 1],
                [2, 2, 2]
            ],
            [
                [3, 3, 3],
                [4, 4, 4]
            ]
        ] )

        MyEntropy = WeightedCrossentropy(
            'testentropy',
            MagicMock(),
            "bin"
        )

        with self.assertRaises( NotImplementedError ):
            MyEntropy( Labels, Predictions )

    def test_it_raises_an_error_if_the_given_predictions_have_not_the_same_rank_as_the_labels( self ):
        Labels = TF.constant( [ [1,2], [4,5],] )
        Predictions = TF.constant( [
            [
                [1, 1, 1],
                [2, 2, 2]
            ],
            [
                [3, 3, 3],
                [4, 4, 4]
            ]
        ] )

        MyEntropy = WeightedCrossentropy(
            'testentropy',
            MagicMock(),
            "bin"
        )

        with self.assertRaises( TypeError ):
            MyEntropy( Labels, Predictions )

    def test_it_raises_an_error_if_the_given_weights_are_incompatible( self ):
        Labels = TF.constant( [
                [1, 1 ],
                [3, 3 ],
        ] )
        Predictions = TF.constant( [
                [1, 1 ],
                [4, 4 ],
        ] )

        self.__NP.ones.return_value = Array( [ [0,0] ] )

        MyEntropy = WeightedCrossentropy(
            'testentropy',
            dict(),
            "bin"
        )

        with self.assertRaises( TypeError ):
            MyEntropy( Labels, Predictions )

    def test_it_hot_encodes_the_predictions( self ):
        Maximum = MagicMock()
        Labels = TF.constant( [
                [1, 1 ],
                [3, 3 ],
        ] )
        Predictions = TF.constant( [
                [1, 1 ],
                [4, 4 ],
        ] )

        Classes = { 0: 1, 1: 2 }

        self.__Adapter.argmax.return_value = Maximum

        MyEntropy = WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        MyEntropy( Labels, Predictions )

        self.__Adapter.argmax.assert_called_once_with(
            Predictions
        )

        self.__Adapter.one_hot.assert_called_once_with(
            Maximum,
            len( Classes )
        )

    def test_it_normalizes_the_axis_of_the_Lables( self ):
        Labels = TF.constant( [
                [1, 1 ],
                [3, 3 ],
        ] )
        Predictions = TF.constant( [
                [1, 1 ],
                [4, 4 ],
        ] )

        Classes = { 0: 1, 1: 2 }

        self.__Adapter.argmax.return_value = MagicMock()

        MyEntropy = WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        MyEntropy( Labels, Predictions )

        Arguments = self.__Adapter.expand_dims.call_args_list[ 1 ][ 0 ]

        self.assertEqual(
            Labels.numpy().tolist(),
            Arguments[ 0 ].numpy().tolist()
        )

        self.assertEqual(
            2,
            Arguments[ 1 ]
        )

    def test_it_casts_the_normalized_Lables( self ):
        Labels = TF.constant( [
                [1, 1 ],
                [3, 3 ],
        ] )
        Predictions = TF.constant( [
                [1, 1 ],
                [4, 4 ],
        ] )

        Classes = { 0: 1, 1: 2 }

        NormLabels = MagicMock()
        NormClasses = MagicMock()
        NormPredictions = MagicMock()
        CastBase = MagicMock()

        Norms = [ NormClasses, NormLabels, NormPredictions ]

        def returnNorm( _, __ ):
            return Norms.pop( 0 )

        self.__Adapter.expand_dims.side_effect = returnNorm
        self.__Adapter.floatx.return_value = CastBase

        MyEntropy = WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        MyEntropy( Labels, Predictions )

        self.__Adapter.cast.assert_any_call(
            NormLabels,
            CastBase
        )


    def test_it_normalizes_the_axis_of_the_hot_encoded_Predictions( self ):
        Labels = TF.constant( [
                [1, 1 ],
                [3, 3 ],
        ] )
        Predictions = TF.constant( [
                [1, 1 ],
                [4, 4 ],
        ] )

        Classes = { 0: 1, 1: 2 }

        HotEncodedPredictions = MagicMock()

        self.__Adapter.argmax.return_value = MagicMock()
        self.__Adapter.one_hot.return_value = HotEncodedPredictions

        MyEntropy = WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        MyEntropy( Labels, Predictions )

        self.__Adapter.expand_dims.assert_any_call(
            HotEncodedPredictions,
            1
        )

    def test_it_casts_the_normalized_Predictions( self ):
        Labels = TF.constant( [
                [1, 1 ],
                [3, 3 ],
        ] )
        Predictions = TF.constant( [
                [1, 1 ],
                [4, 4 ],
        ] )

        Classes = { 0: 1, 1: 2 }

        NormLabels = MagicMock()
        NormClasses = MagicMock()
        NormPredictions = MagicMock()
        CastBase = MagicMock()

        Norms = [ NormClasses, NormLabels, NormPredictions ]

        def returnNorm( _, __ ):
            return Norms.pop( 0 )

        self.__Adapter.expand_dims.side_effect = returnNorm
        self.__Adapter.floatx.return_value = CastBase

        MyEntropy = WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        MyEntropy( Labels, Predictions )

        self.__Adapter.cast.assert_any_call(
            NormPredictions,
            CastBase
        )

    def test_it_normalizes_the_axis_of_the_cost_matrix_on_init( self ):
        Remaped = MagicMock()
        self.__NP.ones.return_value = Remaped

        WeightedCrossentropy(
            'testentropy',
            MagicMock(),
            "bin"
        )

        self.__Adapter.expand_dims.assert_called_once_with(
            Remaped,
            0
        )

    def test_it_casts_the_normalized_Classes_on_init( self ):
        Classes = { 0: 1, 1: 2 }

        NormClasses = MagicMock()
        CastBase = MagicMock()

        self.__Adapter.expand_dims.return_value = NormClasses
        self.__Adapter.floatx.return_value = CastBase

        WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        self.__Adapter.cast.assert_called_with(
            NormClasses,
            CastBase
        )

    def test_it_normalizes_the_axis_of_the_cost_matrix_on_set( self ):
        MyEntropy = WeightedCrossentropy(
            'testentropy',
            MagicMock(),
            "bin"
        )

        New = Array( [ 0 ] )
        MyEntropy.setCostMatrix( New )

        self.__Adapter.expand_dims.assert_called_with(
            New,
            0
        )

    def test_it_casts_the_normalized_Classes_on_set( self ):
        Classes = { 0: 1, 1: 2 }

        NormClasses = MagicMock()
        CastBase = MagicMock()

        self.__Adapter.expand_dims.return_value = NormClasses
        self.__Adapter.floatx.return_value = CastBase

        New = Array( [ 0 ] )
        MyEntropy = WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        MyEntropy.setCostMatrix( New )

        self.__Adapter.cast.assert_called_with(
            NormClasses,
            CastBase
        )


    def test_it_builds_the_product_of_the_normalized_Labels_and_Predictions( self ):
        Labels = TF.constant( [
                [1, 1 ],
                [3, 3 ],
        ] )
        Predictions = TF.constant( [
                [1, 1 ],
                [4, 4 ],
        ] )
        Classes = { 0: 1, 1: 2 }

        NormLabels = MagicMock()
        NormClasses = MagicMock()
        NormPredictions = MagicMock()

        Norms = [ NormClasses, NormLabels, NormPredictions ]

        def returnNorm( _, __ ):
            return Norms.pop( 0 )

        self.__Adapter.cast.side_effect = returnNorm

        MyEntropy = WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        MyEntropy( Labels, Predictions )

        self.__Multiplication.assert_any_call(
            NormLabels,
            NormPredictions
        )

    def test_it_builds_the_product_of_the_normalized_costs_and_the_product_of_Labels_and_Predictions( self ):
        Labels = TF.constant( [
                [1, 1 ],
                [3, 3 ],
        ] )
        Predictions = TF.constant( [
                [1, 1 ],
                [4, 4 ],
        ] )
        Classes = { 0: 1, 1: 2 }

        NormLabels = MagicMock()
        NormClasses = MagicMock()
        NormPredictions = MagicMock()
        ProductOfLabelAndPrediction = MagicMock()

        Norms = [ NormClasses, NormLabels, NormPredictions ]

        def returnNorm( _, __ ):
            return Norms.pop( 0 )

        self.__Adapter.cast.side_effect = returnNorm
        self.__Multiplication.return_value = ProductOfLabelAndPrediction

        MyEntropy = WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        MyEntropy( Labels, Predictions )

        self.__Multiplication.assert_any_call(
            NormClasses,
            ProductOfLabelAndPrediction,
        )

    def test_it_uses_the_sum_of_product_of_the_normalized_inputs( self ):
        Labels = TF.constant( [
                [1, 1 ],
                [3, 3 ],
        ] )
        Predictions = TF.constant( [
                [1, 1 ],
                [4, 4 ],
        ] )
        Classes = { 0: 1, 1: 2 }

        NormLabels = MagicMock()
        NormClasses = MagicMock()
        NormPredictions = MagicMock()
        ProductOfLabelAndPrediction = MagicMock()
        ProductOfAll = MagicMock()

        Norms = [ NormClasses, NormLabels, NormPredictions ]
        Products = [ ProductOfLabelAndPrediction, ProductOfAll ]

        def returnNorm( _, __ ):
            return Norms.pop( 0 )

        def returnProducts( _, __ ):
            return Products.pop( 0 )

        self.__Adapter.cast.side_effect = returnNorm
        self.__Multiplication.side_effect = returnProducts


        MyEntropy = WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        MyEntropy( Labels, Predictions )

        self.__Adapter.sum.assert_called_once_with(
            ProductOfAll,
            axis = [ 1, 2 ]
        )

    def test_it_calls_the_loss_function_with_the_weigths( self ):
        Labels = TF.constant( [
                [1, 1 ],
                [3, 3 ],
        ] )
        Predictions = TF.constant( [
                [1, 1 ],
                [4, 4 ],
        ] )
        Classes = { 0: 1, 1: 2 }

        ComputedWeights = MagicMock()
        WrappedLossFunction = MagicMock(spec=Loss)
        self.__Bin.return_value = WrappedLossFunction

        self.__Adapter.sum.return_value = ComputedWeights


        MyEntropy = WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        MyEntropy( Labels, Predictions )
        Arguments = WrappedLossFunction.call_args_list[ 0 ][ 0 ]

        self.assertEqual(
            Labels.numpy().tolist(),
            Arguments[ 0 ].numpy().tolist(),
        )

        self.assertEqual(
            Predictions.numpy().tolist(),
            Arguments[ 1 ].numpy().tolist(),
        )

        self.assertEqual(
            ComputedWeights,
            Arguments[ 2 ],
        )

    def test_it_returns_the_result_of_the_loss_function( self ):
        Labels = TF.constant( [
                [1, 1 ],
                [3, 3 ],
        ] )
        Predictions = TF.constant( [
                [1, 1 ],
                [4, 4 ],
        ] )
        Classes = { 0: 1, 1: 2 }

        Result = MagicMock()
        self.__Bin.return_value = MagicMock( return_value = Result )
        WrappedLossFunction = MagicMock(spec=Loss)
        WrappedLossFunction.return_value = Result

        MyEntropy = WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        self.assertEqual(
            Result,
            MyEntropy( Labels, Predictions ),
        )

    def test_intis_cost_matrix( self ):
        Classes = { 0: 1, 1: 2 }

        WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        self.__NP.ones.assert_called_once_with(
            ( len( Classes ), len( Classes ) )
        )

    def test_it_maps_given_class_weights( self ):
        Classes = { 0: 1, 1: 2, 2: 3 }
        Remaped = Array( [ [ 0, 0, 0 ], [ 0, 0, 0 ], [ 0, 0, 0 ] ] )

        self.__NP.ones.return_value = Remaped
        WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        self.assertEqual(
            [[1, 2, 3], [2, 0, 0], [3, 0, 0]],
            Remaped.tolist()
        )

    def test_it_returns_the_current_configuration( self ):
        Name = 'testentropy'
        EntropyKey = "bin"
        Classes = { 0: 1, 1: 2, 2: 3 }
        name = "something"

        self.__NP.ones.return_value = Array( [ [0,0,0],[0,0,0], [0,0,0 ] ] )
        MyEntropy = WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin",
            { 'name': name }
        )

        self.assertDictEqual(
            {
                'Name': Name,
                'ClassWeights': [[1, 2, 3], [2, 0, 0], [3, 0, 0]],
                'EntropyKey': EntropyKey,
                'Reduction': None,
                'KeywordedEntropyArgs': { 'name': name },
            },
            MyEntropy.get_config()
        )


    def test_it_accetps_lists_as_weights( self ):
        Classes = [ [1, 2, 3], [2, 0, 0], [3, 0, 0]]

        self.__NP.array = MagicMock( return_value = Array( Classes ) )

        MyEntropy = WeightedCrossentropy(
            'testentropy',
            Classes,
            "bin"
        )

        self.assertListEqual(
            Classes,
            MyEntropy.get_config()[ 'ClassWeights' ]
        )
