package de.huberlin.biomed.pipeline

import edu.stanford.nlp.pipeline.CoreDocument
import edu.stanford.nlp.pipeline.CoreSentence
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import edu.stanford.nlp.util.logging.RedwoodConfiguration
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mockito
import org.mockito.Mockito.*
import org.powermock.api.mockito.PowerMockito.*
import org.powermock.core.classloader.annotations.PrepareForTest
import org.powermock.modules.junit4.PowerMockRunner
import java.util.*
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@PrepareForTest(
	StanfordCoreNLP::class,
	RedwoodConfiguration::class,
	StanfordNLPPipe::class
)
@RunWith( PowerMockRunner::class )
class StanfordNLPPipeSpec
{
	private lateinit var NLPPipeline: StanfordCoreNLP
	private lateinit var Logger: RedwoodConfiguration
	private lateinit var Doc: CoreDocument

	@Before
	fun setup()
	{
		this.NLPPipeline = Mockito.mock( StanfordCoreNLP::class.java )
		this.Logger = Mockito.mock( RedwoodConfiguration::class.java )
		this.Doc = Mockito.mock( CoreDocument::class.java )

		mockStatic( RedwoodConfiguration::class.java )

		whenNew( StanfordCoreNLP::class.java ).withAnyArguments().thenReturn( this.NLPPipeline )
		Mockito.`when`( RedwoodConfiguration.errorLevel() ).thenReturn( this.Logger )

		whenNew( CoreDocument::class.java ).withAnyArguments().thenReturn( this.Doc )
	}

	@Test
	fun `it is a Pipeline`()
	{
		val MyPipe: Any = StanfordNLPPipe.getInstance()
		assertTrue( MyPipe is Pipeline )
	}

	@Test
	fun `it lowers the loglevel`()
	{
		StanfordNLPPipe.getInstance()
		verifyStatic( RedwoodConfiguration::class.java, times( 1 ) )
		RedwoodConfiguration.errorLevel()

		verify( this.Logger, times( 1 ) ).apply()

	}

	@Test
	fun `it initializes the Pipeline`()
	{
		val Given = Properties()
		Given.setProperty( "threads", "1" )
		Given.setProperty(
			"annotators", "tokenize,ssplit,pos,lemma"
		)
		whenNew( StanfordCoreNLP::class.java )
			.withArguments( Given )
			.thenReturn( this.NLPPipeline )

		StanfordNLPPipe.getInstance()

		verifyNew( StanfordCoreNLP::class.java ).withArguments( Given )
	}

	@Test
	fun `it initializes the document`()
	{
		val GivenText = "I am a text message"
		val MyNLP = StanfordNLPPipe.getInstance()
		whenNew( CoreDocument::class.java )
			.withArguments( GivenText )
			.thenReturn( this.Doc )

		MyNLP.apply( GivenText )

		verifyNew( CoreDocument::class.java ).withArguments( GivenText )
	}

	@Test
	fun `it annotates the document`()
	{
		val MyNLP = StanfordNLPPipe.getInstance()

		MyNLP.apply( "I am a text message" )

		verify( this.NLPPipeline, times( 1 ) ).annotate( this.Doc )
	}

	@Test
	fun `it returns a list of sentences`()
	{
		val Expected = listOf<CoreSentence>()
		Mockito.`when`( this.Doc.sentences() ).thenReturn( Expected )

		val MyNLP = StanfordNLPPipe.getInstance()
		assertEquals(
			expected = Expected,
			actual = MyNLP.apply( "I am a text message" )
		)
	}
}