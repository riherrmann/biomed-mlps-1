package de.huberlin.biomed.processor

import de.huberlin.biomed.filter.BiomedFilter
import de.huberlin.biomed.filter.Filter
import de.huberlin.biomed.filter.FilterFactory
import de.huberlin.biomed.pipeline.Pipeline
import de.huberlin.biomed.pipeline.PipelineFactory
import de.huberlin.biomed.pipeline.StanfordNLPPipe
import edu.stanford.nlp.pipeline.CoreSentence

internal class BiomedProcessor private constructor(
	private val Pipeline: Pipeline,
	private val Filters: Filter
): Processor
{
	override fun process(
		Document: String
	): String = this.transformBack(
		this.applyFilters(
			this.Pipeline.apply( Document )
		)
	)

	private fun applyFilters( Sentences: List<CoreSentence> ): List<List<String>>
	{
		val Collector = mutableListOf<List<String>>()
		for( Sentence in Sentences )
		{
			Collector.add( this.Filters.filter( Sentence ) )
		}

		return Collector
	}

	private fun transformBack( Sentences: List<List<String>>): String
	{
		val Collector = StringBuilder()

		for( Sentence in Sentences )
		{
			for( Word in Sentence )
			{
				Collector.append( Word )
				Collector.append( " " )
			}
		}

		return Collector.toString().trim()
	}

	companion object Factory: ProcessorFactory
	{
		private val PipelineFactory: PipelineFactory = StanfordNLPPipe
		private val FilterFactory: FilterFactory = BiomedFilter

		override fun getInstance(
			Flags: String
		): Processor = BiomedProcessor(
			this.PipelineFactory.getInstance(),
			this.FilterFactory.getInstance( Flags )
		)
	}
}