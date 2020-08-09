package de.huberlin.biomed.processor

import de.huberlin.biomed.filter.Filter
import de.huberlin.biomed.filter.FilterFactory
import de.huberlin.biomed.pipeline.Pipeline
import de.huberlin.biomed.pipeline.PipelineFactory
import edu.stanford.nlp.pipeline.CoreSentence
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.jupiter.api.Assertions.assertTrue
import org.mockito.Mockito.*
import org.powermock.reflect.Whitebox
import kotlin.test.assertEquals

class BiomedProcessorSpec
{
	private lateinit var PF: StubbedPipelineFactory
	private lateinit var FF: StubbedFilterFactory
	private lateinit var P: Pipeline
	private lateinit var F: Filter

	private lateinit var ResetPF: PipelineFactory
	private lateinit var ResetFF: FilterFactory

	@Before
	fun setup()
	{
		this.P = mock( Pipeline::class.java )
		this.F = mock( Filter::class.java )

		this.PF = StubbedPipelineFactory()
		this.PF.GivenInstance = this.P
		this.FF = StubbedFilterFactory()
		this.FF.GivenInstance = this.F

		this.ResetPF = Whitebox.getInternalState( BiomedProcessor::class.java, "PipelineFactory" )
		this.ResetFF = Whitebox.getInternalState( BiomedProcessor::class.java, "FilterFactory" )

		`when`( this.P.apply( anyString() ) ).thenReturn( listOf() )

		Whitebox.setInternalState( BiomedProcessor::class.java, "PipelineFactory", this.PF )
		Whitebox.setInternalState( BiomedProcessor::class.java, "FilterFactory", this.FF )
	}

	@After
	fun tearDown()
	{
		Whitebox.setInternalState( BiomedProcessor::class.java, "PipelineFactory", this.ResetPF )
		Whitebox.setInternalState( BiomedProcessor::class.java, "FilterFactory", this.ResetFF )
	}

	@Test
	fun `it is a Processor`()
	{
		val MyProcessor: Any = BiomedProcessor.getInstance( "na" )
		assertTrue( MyProcessor is Processor )
	}

	@Test
	fun `it initializes the NLPPipeline`()
	{
		BiomedProcessor.getInstance( "na" )
		assertEquals(
			actual = this.PF.CallCounter,
			expected = 1
		)
	}

	@Test
	fun `it initializes the Filters`()
	{
		val Flags = "na"
		BiomedProcessor.getInstance( Flags )
		assertEquals(
			actual = this.FF.CallCounter,
			expected = 1
		)
		assertEquals(
			actual = this.FF.CapturedFlags,
			expected = Flags
		)
	}

	@Test
	fun `it calls the Pipeline, with a given text`()
	{
		val MyDocument = "I am simple text."

		val MyProcessor = BiomedProcessor.getInstance( "na" )
		MyProcessor.process( MyDocument )

		verify( this.P, times( 1 ) ).apply( MyDocument )
	}

	@Test
	fun `it delegates the parsed sentences to the filter`()
	{
		val S1 = mock( CoreSentence::class.java )
		val S2 = mock( CoreSentence::class.java )
		val S3 = mock( CoreSentence::class.java )
		val Parsed = listOf( S1, S2, S3 )

		`when`( this.P.apply( anyString() ) ).thenReturn( Parsed )

		val MyProcessor = BiomedProcessor.getInstance( "na" )
		MyProcessor.process( "I am simple text." )

		verify( this.F, times( 1 ) ).filter( S1 )
		verify( this.F, times( 1 ) ).filter( S2 )
		verify( this.F, times( 1 ) ).filter( S3 )
	}

	@Test
	fun `it returns the filtered document`()
	{
		val S1 = mock( CoreSentence::class.java )
		val S2 = mock( CoreSentence::class.java )
		val Parsed = listOf( S1, S2 )

		`when`( this.P.apply( anyString() ) ).thenReturn( Parsed )
		`when`( this.F.filter( S1 ) ).thenReturn( listOf( "A", "B" ) )
		`when`( this.F.filter( S2 ) ).thenReturn( listOf( "C", "D" ) )

		val MyProcessor = BiomedProcessor.getInstance( "na" )
		assertEquals(
			actual = MyProcessor.process( "I am simple text." ),
			expected =  "A B C D"
		)
	}

	private class StubbedPipelineFactory: PipelineFactory
	{
		var CallCounter = 0
		lateinit var GivenInstance: Pipeline

		override fun getInstance(): Pipeline = this.GivenInstance.also { this.CallCounter++ }
	}

	private class StubbedFilterFactory: FilterFactory
	{
		var CallCounter = 0
		lateinit var CapturedFlags: String
		lateinit var GivenInstance: Filter

		override fun getInstance(
			Flags: String
		): Filter = this.GivenInstance.also {
			this.CallCounter++
			this.CapturedFlags = Flags
		}
	}
}