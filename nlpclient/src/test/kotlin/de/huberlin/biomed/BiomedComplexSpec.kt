package de.huberlin.biomed

import de.huberlin.biomed.io.BiomedInteractor
import de.huberlin.biomed.io.Interactor
import de.huberlin.biomed.processor.BiomedProcessor
import de.huberlin.biomed.processor.Processor
import de.huberlin.biomed.processor.ProcessorFactory
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.ArgumentMatchers.anyString
import org.mockito.Mockito.*
import org.powermock.core.classloader.annotations.PrepareForTest
import org.powermock.modules.junit4.PowerMockRunner
import org.powermock.reflect.Whitebox
import kotlin.test.assertTrue

@PrepareForTest(
	BiomedInteractor::class,
	BiomedProcessor::class,
	BiomedComplex::class
)
@RunWith( PowerMockRunner::class )
class BiomedComplexSpec
{
	private lateinit var Interactions: Interactor
	private lateinit var ProcF: ProcessorFactory
	private lateinit var Proc: Processor

	@Before
	fun setup()
	{
		this.Interactions = mock( Interactor::class.java )
		this.ProcF = mock( ProcessorFactory::class.java )
		this.Proc = mock( Processor::class.java )

		`when`( this.ProcF.getInstance( anyString() ) ).thenReturn( this.Proc )
		`when`( this.Interactions.getFlags() ).thenReturn( "something" )

		Whitebox.setInternalState( BiomedComplex::class.java, "ProcessFactory", this.ProcF )
	}

	@After
	fun tearDown()
	{
		Whitebox.setInternalState( BiomedComplex::class.java, "ProcessFactory", BiomedProcessor )
	}

	@Test
	fun `it is a instance of Complex`()
	{
		val MyNormalizer: Any = BiomedComplex.getInstance( this.Interactions )
		assertTrue( MyNormalizer is Complex )
	}

	@Test
	fun `it initializes the Processor`()
	{
		val ExpectedFlags = "na"
		this.Interactions = mock( Interactor::class.java )
		`when`( this.Interactions.getFlags() ).thenReturn( ExpectedFlags )
		`when`( this.ProcF.getInstance( ExpectedFlags ) ).thenReturn( this.Proc )

		BiomedComplex.getInstance( this.Interactions )

		verify( this.ProcF, times( 1 ) ).getInstance( ExpectedFlags )
	}

	@Test
	fun `it delegates the documents to the Processor until none is left`()
	{
		val D1 = "Doc1"
		val D2 = "Doc2"
		val D3 = "Doc3"
		`when`( this.Interactions.getNextDocument() )
			.thenReturn( D1 )
			.thenReturn( D2 )
			.thenReturn( D3 )
			.thenReturn( null )

		val MyNormalizer = BiomedComplex.getInstance( this.Interactions )
		MyNormalizer.run()

		verify( this.Interactions, times( 4 ) ).getNextDocument()
		verify( this.Proc, times( 3 ) ).process( anyString() )
		verify( this.Proc, times( 1 ) ).process( D1 )
		verify( this.Proc, times( 1 ) ).process( D2 )
		verify( this.Proc, times( 1 ) ).process( D3 )
	}

	@Test
	fun `it writes back the results of the processing`()
	{
		val D1 = "Doc1"
		val R1 = "Res1"
		val D2 = "Doc2"
		val R2 = "Res2"
		val D3 = "Doc3"
		val R3 = "Res3"
		`when`( this.Interactions.getNextDocument() )
			.thenReturn( D1 )
			.thenReturn( D2 )
			.thenReturn( D3 )
			.thenReturn( null )

		`when`( this.Proc.process( D1 ) ).thenReturn( R1 )
		`when`( this.Proc.process( D2 ) ).thenReturn( R2 )
		`when`( this.Proc.process( D3 ) ).thenReturn( R3 )

		val MyNormalizer = BiomedComplex.getInstance( this.Interactions )
		MyNormalizer.run()

		verify( this.Interactions, times( 3 ) ).returnResult( anyString() )
		verify( this.Interactions, times( 1 ) ).returnResult( R1 )
		verify( this.Interactions, times( 1 ) ).returnResult( R2 )
		verify( this.Interactions, times( 1 ) ).returnResult( R3 )
	}
}