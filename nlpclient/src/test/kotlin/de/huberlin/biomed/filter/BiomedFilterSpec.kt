package de.huberlin.biomed.filter

import edu.stanford.nlp.pipeline.CoreSentence
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.mockito.Mockito.`when`
import org.mockito.Mockito.mock
import kotlin.test.assertEquals

class BiomedFilterSpec
{
	private lateinit var Sentence: CoreSentence

	@Before
	fun setUp()
	{
		this.Sentence = mock( CoreSentence::class.java )
	}

	@Test
	fun `it is a Filter`()
	{
		val MyFilter: Any = BiomedFilter.getInstance( "n" )
		assertTrue( MyFilter is Filter )
	}

	@Test
	fun `it filters nouns`()
	{
		val Lemmas = listOf(
			"pancreatic",
			"cancer",
			"be",
			"the",
			"most",
			"lethal",
			"of",
			"all",
			"solid",
			"tumor",
			"partially",
			"because",
			"of",
			"its",
			"chemoresistance",
			"."
		)

		val Forms = listOf(
			"JJ",
			"NNS",
			"VBZ",
			"DT",
			"RBS",
			"JJ",
			"IN",
			"DT",
			"JJ",
			"NNP",
			"RB",
			"IN",
			"IN",
			"PRP$",
			"NN",
			"."
		)

		`when`( this.Sentence.lemmas() ).thenReturn( Lemmas )
		`when`( this.Sentence.posTags() ).thenReturn( Forms )

		val MyFilter = BiomedFilter.getInstance( "n" )

		assertEquals(
			expected = listOf(
				"cancer",
				"tumor",
				"chemoresistance"
			),
			actual = MyFilter.filter( this.Sentence )
		)
	}

	@Test
	fun `it filters verbs`()
	{
		val Lemmas = listOf(
			"pancreatic",
			"cancer",
			"be",
			"the",
			"most",
			"lethal",
			"of",
			"all",
			"solid",
			"tumor",
			"partially",
			"because",
			"of",
			"its",
			"chemoresistance",
			"."
		)

		val Forms = listOf(
			"JJ",
			"NNS",
			"VBZ",
			"DT",
			"RBS",
			"JJ",
			"IN",
			"DT",
			"MD",
			"NNP",
			"RB",
			"IN",
			"IN",
			"PRP$",
			"NN",
			"."
		)

		`when`( this.Sentence.lemmas() ).thenReturn( Lemmas )
		`when`( this.Sentence.posTags() ).thenReturn( Forms )

		val MyFilter = BiomedFilter.getInstance( "v" )

		assertEquals(
			expected = listOf(
				"be",
				"solid"
			),
			actual = MyFilter.filter( this.Sentence )
		)
	}

	@Test
	fun `it filters adjectives`()
	{
		val Lemmas = listOf(
			"pancreatic",
			"cancer",
			"be",
			"the",
			"most",
			"lethal",
			"of",
			"all",
			"solid",
			"tumor",
			"partially",
			"because",
			"of",
			"its",
			"chemoresistance",
			"."
		)

		val Forms = listOf(
			"JJ",
			"NNS",
			"VBZ",
			"DT",
			"RBS",
			"JJ",
			"IN",
			"DT",
			"JJ",
			"NNP",
			"RB",
			"IN",
			"IN",
			"PRP$",
			"NN",
			"."
		)

		`when`( this.Sentence.lemmas() ).thenReturn( Lemmas )
		`when`( this.Sentence.posTags() ).thenReturn( Forms )

		val MyFilter = BiomedFilter.getInstance( "a" )

		assertEquals(
			expected = listOf(
				"pancreatic",
				"lethal",
				"solid"
			),
			actual = MyFilter.filter( this.Sentence )
		)
	}

	@Test
	fun `it filters adverbs`()
	{
		val Lemmas = listOf(
			"pancreatic",
			"cancer",
			"be",
			"the",
			"most",
			"lethal",
			"of",
			"all",
			"solid",
			"tumor",
			"partially",
			"because",
			"of",
			"its",
			"chemoresistance",
			"."
		)

		val Forms = listOf(
			"JJ",
			"NNS",
			"VBZ",
			"DT",
			"RBS",
			"JJ",
			"IN",
			"DT",
			"JJ",
			"NNP",
			"WRB",
			"IN",
			"IN",
			"PRP$",
			"NN",
			"."
		)

		`when`( this.Sentence.lemmas() ).thenReturn( Lemmas )
		`when`( this.Sentence.posTags() ).thenReturn( Forms )

		val MyFilter = BiomedFilter.getInstance( "d" )

		assertEquals(
			expected = listOf(
				"most",
				"partially"
			),
			actual = MyFilter.filter( this.Sentence )
		)
	}

	@Test
	fun `it filters numerals`()
	{
		val Lemmas = listOf(
			"pancreatic",
			"cancer",
			"be",
			"the",
			"1st",
			"lethal",
			"of",
			"all",
			"solid",
			"tumor",
			"2",
			"because",
			"of",
			"its",
			"chemoresistance",
			"."
		)

		val Forms = listOf(
			"NNS",
			"NNS",
			"VBZ",
			"DT",
			"CD",
			"JJ",
			"IN",
			"DT",
			"JJ",
			"NNP",
			"CC",
			"IN",
			"IN",
			"PRP$",
			"NN",
			"."
		)

		`when`( this.Sentence.lemmas() ).thenReturn( Lemmas )
		`when`( this.Sentence.posTags() ).thenReturn( Forms )

		val MyFilter = BiomedFilter.getInstance( "i" )

		assertEquals(
			expected = listOf(
				"1s",
				"2"
			),
			actual = MyFilter.filter( this.Sentence )
		)
	}

	@Test
	fun `it filters symbols`()
	{
		val Lemmas = listOf(
			"pancreatic",
			"cancer",
			"be",
			"the",
			"1st",
			"lethal",
			"of",
			"all",
			"solid",
			"tumor",
			"2",
			"because",
			"of",
			"its",
			"chemoresistance",
			"."
		)

		val Forms = listOf(
			"SYM",
			"NNS",
			"VBZ",
			"DT",
			"CD",
			"JJ",
			"IN",
			"DT",
			"JJ",
			"NNP",
			"CC",
			"IN",
			"IN",
			"PRP$",
			"NN",
			"."
		)

		`when`( this.Sentence.lemmas() ).thenReturn( Lemmas )
		`when`( this.Sentence.posTags() ).thenReturn( Forms )

		val MyFilter = BiomedFilter.getInstance( "y" )

		assertEquals(
			expected = listOf(
				"pancreatic"
			),
			actual = MyFilter.filter( this.Sentence )
		)
	}

	@Test
	fun `it always adds foreign words`()
	{
		val Lemmas = listOf(
			"pancreatic",
			"cancer",
			"be",
			"the",
			"most",
			"lethal",
			"of",
			"all",
			"solid",
			"tumor",
			"partially",
			"because",
			"of",
			"its",
			"chemoresistance",
			"."
		)

		val Forms = listOf(
			"JJ",
			"NNS",
			"VBZ",
			"DT",
			"RBS",
			"JJ",
			"IN",
			"DT",
			"JJ",
			"NNP",
			"RB",
			"IN",
			"IN",
			"PRP$",
			"FW",
			"."
		)

		`when`( this.Sentence.lemmas() ).thenReturn( Lemmas )
		`when`( this.Sentence.posTags() ).thenReturn( Forms )

		val MyFilter = BiomedFilter.getInstance( "n" )

		assertEquals(
			expected = listOf(
				"cancer",
				"tumor",
				"chemoresistance"
			),
			actual = MyFilter.filter( this.Sentence )
		)
	}

	@Test
	fun `it adds mixed content according to the flags`()
	{
		val Lemmas = listOf(
			"pancreatic",
			"cancer",
			"be",
			"the",
			"most",
			"lethal",
			"of",
			"all",
			"solid",
			"tumor",
			"partially",
			"because",
			"of",
			"its",
			"chemoresistance",
			"."
		)

		val Forms = listOf(
			"JJ",
			"NNS",
			"VBZ",
			"DT",
			"RBS",
			"JJ",
			"IN",
			"DT",
			"JJ",
			"NNP",
			"RB",
			"IN",
			"IN",
			"PRP$",
			"FW",
			"."
		)

		`when`( this.Sentence.lemmas() ).thenReturn( Lemmas )
		`when`( this.Sentence.posTags() ).thenReturn( Forms )

		val MyFilter = BiomedFilter.getInstance( "na" )

		assertEquals(
			expected = listOf(
				"pancreatic",
				"cancer",
				"lethal",
				"solid",
				"tumor",
				"chemoresistance"
			),
			actual = MyFilter.filter( this.Sentence )
		)
	}
}