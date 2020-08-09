import unittest
from unittest.mock import MagicMock, patch
from biomed.encoder.std_categorie_encoder import StdCategoriesEncoder
from biomed.encoder.categorie_encoder import CategoriesEncoder
from pandas import Series
from numpy import array as Array

class StdCategoriesEncoderSpec( unittest.TestCase ):
    def setUp( self ):
        self.__EncoderF = patch( 'biomed.encoder.std_categorie_encoder.Encoder' )
        self.__Encoder = self.__EncoderF.start()
        self.__Encoder.return_value = self.__Encoder

    def tearDown( self ):
        self.__EncoderF.stop()

    def test_it_is_a_encoder( self ):
        MyEncoder = StdCategoriesEncoder.Factory.getInstance()
        self.assertTrue( isinstance( MyEncoder, CategoriesEncoder ) )

    def test_it_sets_given_categories( self ):
        Categories = Series( [ 4, 1, 1, 2, 3 ] )

        MyEncoder = StdCategoriesEncoder.Factory.getInstance()
        MyEncoder.setCategories( Categories )

        Arguments, _ = self.__Encoder.fit.call_args_list[ 0 ]

        self.assertListEqual(
            [ 1, 2, 3, 4 ],
            Arguments[ 0 ].tolist()
        )

        self.__Encoder.fit.assert_called_once()

    def test_it_fails_to_return_the_categories_if_no_categories_had_been_set( self ):
        MyEncoder = StdCategoriesEncoder.Factory.getInstance()

        with self.assertRaises( RuntimeError, msg = "No Categories had been fit in so far" ):
            MyEncoder.getCategories()

    def test_it_returns_the_categories( self ):
        self.__Encoder.classes_ = Array( [ 1, 2, 3, 4 ] )

        MyEncoder = StdCategoriesEncoder.Factory.getInstance()
        MyEncoder.setCategories( Series( [ 4, 1, 1, 2, 3 ] ) )

        self.assertListEqual(
            [ 1, 2, 3, 4 ],
            MyEncoder.getCategories()
        )

    def test_it_fails_to_return_the_amount_of_categories_if_no_categories_had_been_set( self ):
        MyEncoder = StdCategoriesEncoder.Factory.getInstance()

        with self.assertRaises( RuntimeError, msg = "No Categories had been fit in so far" ):
            MyEncoder.amountOfCategories()

    def test_it_returns_the_amount_of_categories( self ):
        self.__Encoder.classes_ = Array( [ 1, 2, 3, 4 ] )

        MyEncoder = StdCategoriesEncoder.Factory.getInstance()
        MyEncoder.setCategories( Series( [ 4, 1, 1, 2, 3 ] ) )

        self.assertEqual(
            4,
            MyEncoder.amountOfCategories()
        )

    def test_it_fails_to_return_encoded_labels_if_no_categories_had_been_set( self ):
        MyEncoder = StdCategoriesEncoder.Factory.getInstance()

        with self.assertRaises( RuntimeError, msg = "No Categories had been fit in so far" ):
            MyEncoder.encode( Array( [ 1, 2, 3 ] ) )

    def test_it_encodes_given_labels( self ):
        Labels = Array( [ 1, 1, 2 ] )
        ExpectedReturn = MagicMock()

        self.__Encoder.transform.return_value = ExpectedReturn

        MyEncoder = StdCategoriesEncoder.Factory.getInstance()
        MyEncoder.setCategories( Series( [ 1, 2 ,3 ] ) )

        self.assertEqual(
            ExpectedReturn,
            MyEncoder.encode( Labels )
        )


        Arguments, _ = self.__Encoder.transform.call_args_list[ 0 ]

        self.assertListEqual(
            Labels.tolist(),
            Arguments[ 0 ].tolist()
        )

        self.__Encoder.transform.assert_called_once()

    def test_it_fails_to_return_hotencode_labels_if_no_categories_had_been_set( self ):
        MyEncoder = StdCategoriesEncoder.Factory.getInstance()

        with self.assertRaises( RuntimeError, msg = "No Categories had been fit in so far" ):
            MyEncoder.hotEncode( Array( [ 1, 2, 3 ] ) )

    @patch( 'biomed.encoder.std_categorie_encoder.hotEncode' )
    def test_it_hotencodes_given_labels( self, HotEncoder: MagicMock ):
        Labels = Array( [ 1, 1, 2 ] )
        Encoded = MagicMock()
        ExpectedReturn = MagicMock()

        HotEncoder.return_value = ExpectedReturn
        self.__Encoder.transform.return_value = Encoded
        self.__Encoder.classes_ = Array( [ 1, 2 ] )

        MyEncoder = StdCategoriesEncoder.Factory.getInstance()
        MyEncoder.setCategories( Series( [ 1, 2 ,3 ] ) )

        self.assertEqual(
            ExpectedReturn,
            MyEncoder.hotEncode( Labels )
        )

        HotEncoder.assert_called_once_with(
            Encoded,
            2
        )

    def test_it_fails_to_return_decoded_data_if_no_categories_had_been_set( self ):
        MyEncoder = StdCategoriesEncoder.Factory.getInstance()

        with self.assertRaises( RuntimeError, msg = "No Categories had been fit in so far" ):
            MyEncoder.decode( Array( [ 1, 2, 3 ] ) )

    def test_it_decodes_given_labels( self ):
        Data = Array( [ 0, 1, 0 ] )
        ExpectedReturn = MagicMock()

        self.__Encoder.inverse_transform.return_value = ExpectedReturn

        MyEncoder = StdCategoriesEncoder.Factory.getInstance()
        MyEncoder.setCategories( Series( [ 1, 2 ,3 ] ) )

        self.assertEqual(
            ExpectedReturn,
            MyEncoder.decode( Data )
        )

        Arguments, _ = self.__Encoder.inverse_transform.call_args_list[ 0 ]

        self.assertListEqual(
            Data.tolist(),
            Arguments[ 0 ].tolist()
        )

        self.__Encoder.inverse_transform.assert_called_once()
