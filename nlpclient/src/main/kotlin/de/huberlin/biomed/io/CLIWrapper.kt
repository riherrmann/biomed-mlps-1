package de.huberlin.biomed.io

import org.apache.commons.cli.*

internal object CLIWrapper: CLI
{
	override fun getOptions(): Options = Options()
	override fun getOptionBuilder(): Option.Builder = Option.builder()
	override fun getOptionBuilder( Name: String ): Option.Builder = Option.builder( Name )
	override fun getParser(): CommandLineParser = DefaultParser()
	override fun getHelpFormatter(): HelpFormatter = HelpFormatter()
}