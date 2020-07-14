package de.huberlin.biomed.pipeline

import edu.stanford.nlp.pipeline.CoreDocument
import edu.stanford.nlp.pipeline.CoreSentence
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import edu.stanford.nlp.util.logging.RedwoodConfiguration
import java.util.*

internal class StanfordNLPPipe private constructor(
	private val Pipeline: StanfordCoreNLP
): Pipeline
{
	override fun apply( Document: String ): List<CoreSentence>
	{
		val Doc = CoreDocument( Document )
		this.Pipeline.annotate( Doc )
		return Doc.sentences()
	}

	companion object Factory: PipelineFactory
	{
		private fun resetLogLevel(): Unit = RedwoodConfiguration.errorLevel().apply()

		override fun getInstance(): Pipeline
		{
			this.resetLogLevel()

			val Properties = Properties()
			Properties.setProperty( "threads", "1" )
			Properties.setProperty( "annotators", "tokenize,ssplit,pos,lemma" )

			return StanfordNLPPipe( StanfordCoreNLP( Properties ) )
		}
	}
}