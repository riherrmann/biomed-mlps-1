package de.huberlin.biomed.pipeline

import edu.stanford.nlp.pipeline.CoreSentence

interface Pipeline
{
	fun apply( Document: String ): List<CoreSentence>
}

interface PipelineFactory
{
	fun getInstance(): Pipeline
}