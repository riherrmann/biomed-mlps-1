package de.huberlin.biomed.io

import org.apache.commons.cli.CommandLineParser
import org.apache.commons.cli.HelpFormatter
import org.apache.commons.cli.Option
import org.apache.commons.cli.Options

internal interface CLI
{
	fun getOptions(): Options
	fun getOptionBuilder(): Option.Builder
	fun getOptionBuilder( Name: String ): Option.Builder
	fun getParser(): CommandLineParser
	fun getHelpFormatter(): HelpFormatter
}