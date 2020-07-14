package de.huberlin.biomed.filter

import edu.stanford.nlp.pipeline.CoreSentence

// see: https://catalog.ldc.upenn.edu/docs/LDC99T42/tagguid1.pdf
internal class BiomedFilter private constructor(
	private val UseNouns: Boolean,
	private val UseVerbs: Boolean,
	private val UseAdjectives: Boolean,
	private val UseAdverbs: Boolean
): Filter
{
	override fun filter( Sentence: CoreSentence ): List<String>
	{
		val Lemmas = Sentence.lemmas()
		val Forms = Sentence.posTags()
		val Filtered = mutableListOf<String>()

		for( Index in 0..Lemmas.lastIndex )
		{
			this.filterNouns(
				Index,
				Filtered,
				Lemmas,
				Forms
			)

			this.filterVerbs(
				Index,
				Filtered,
				Lemmas,
				Forms
			)

			this.filterAdjectives(
				Index,
				Filtered,
				Lemmas,
				Forms
			)

			this.filterAdverbs(
				Index,
				Filtered,
				Lemmas,
				Forms
			)

			this.filterForeignWords(
				Index,
				Filtered,
				Lemmas,
				Forms
			)
		}

		return Filtered
	}

	private fun filterAndAppend(
		Position: Int,
		BagOfWords: MutableList<String>,
		Lemmas: List<String>,
		Condition: Boolean
	)
	{
		if( Condition )	{ BagOfWords.add( Lemmas[ Position ] ) }
	}

	private fun filterNouns(
		Position: Int,
		BagOfWords: MutableList<String>,
		Lemmas: List<String>,
		Forms: List<String>
	)
	{
		this.filterAndAppend(
			Position,
			BagOfWords,
			Lemmas,
			this.UseNouns && Forms[ Position ].startsWith( "NN" )
		)
	}

	private fun filterVerbs(
		Position: Int,
		BagOfWords: MutableList<String>,
		Lemmas: List<String>,
		Forms: List<String>
	)
	{
		this.filterAndAppend(
			Position,
			BagOfWords,
			Lemmas,
			this.UseVerbs
			&& ( Forms[ Position ].startsWith( "VB" ) || Forms[ Position ] == "MD" )
		)
	}

	private fun filterAdjectives(
		Position: Int,
		BagOfWords: MutableList<String>,
		Lemmas: List<String>,
		Forms: List<String>
	)
	{
		this.filterAndAppend(
			Position,
			BagOfWords,
			Lemmas,
			this.UseAdjectives && Forms[ Position ].startsWith( "JJ" )
		)
	}

	private fun filterAdverbs(
		Position: Int,
		BagOfWords: MutableList<String>,
		Lemmas: List<String>,
		Forms: List<String>
	)
	{
		this.filterAndAppend(
			Position,
			BagOfWords,
			Lemmas,
			this.UseAdverbs
			&& ( Forms[ Position ].startsWith( "RB" ) || Forms[ Position ] == "WRB" )
		)
	}

	private fun filterForeignWords(
		Position: Int,
		BagOfWords: MutableList<String>,
		Lemmas: List<String>,
		Forms: List<String>
	)
	{
		this.filterAndAppend(
			Position,
			BagOfWords,
			Lemmas,
			Forms[ Position ].startsWith( "FW" )
		)
	}

	companion object Factory: FilterFactory
	{
		private fun useNouns( Flags: String ) = 'n' in Flags
		private fun useVerbs( Flags: String ) = 'v' in Flags
		private fun useAdjectives( Flags: String ) = 'a' in Flags
		private fun useAdverbs( Flags: String ) = 'd' in Flags

		override fun getInstance( Flags: String ): Filter
		{
			val NormFlags = Flags.toLowerCase()

			return BiomedFilter(
				UseNouns = this.useNouns( NormFlags ),
				UseVerbs = this.useVerbs( NormFlags ),
				UseAdjectives = this.useAdjectives( NormFlags ),
				UseAdverbs = this.useAdverbs( NormFlags )
			)
		}
	}
}
