package de.huberlin.biomed.processor

interface Processor
{
	fun process( Document: String ): String
}

interface ProcessorFactory
{
	fun getInstance( Flags: String ): Processor
}