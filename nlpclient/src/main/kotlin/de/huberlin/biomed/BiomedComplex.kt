package de.huberlin.biomed

import de.huberlin.biomed.io.Interactor
import de.huberlin.biomed.processor.BiomedProcessor
import de.huberlin.biomed.processor.Processor
import de.huberlin.biomed.processor.ProcessorFactory

internal class BiomedComplex private constructor(
	private val Interactions: Interactor,
	private val Processor: Processor
): Complex
{
	override fun run()
	{
		var Document: String? = this.Interactions.getNextDocument()

		while( Document != null )
		{
			this.processAndWriteBack( Document )
			Document = this.Interactions.getNextDocument()
		}
	}

	private fun processAndWriteBack(
		Document: String
	): Unit = this.Interactions.returnResult( this.Processor.process( Document ) )

	companion object Factory: ComplexFactory
	{
		private val ProcessFactory: ProcessorFactory = BiomedProcessor

		override fun getInstance(
			Interaction: Interactor
		): Complex = BiomedComplex(
			Interaction,
			this.ProcessFactory.getInstance( Interaction.getFlags() )
		)
	}
}
