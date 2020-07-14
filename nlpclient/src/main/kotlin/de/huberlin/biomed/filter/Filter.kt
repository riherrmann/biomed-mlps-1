package de.huberlin.biomed.filter

import edu.stanford.nlp.pipeline.CoreSentence

interface Filter
{
	fun filter( Sentence: CoreSentence ): List<String>
}

interface FilterFactory
{
	fun getInstance( Flags: String ): Filter
}