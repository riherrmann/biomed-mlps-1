package de.huberlin.biomed.io

import org.apache.commons.cli.CommandLine
import org.apache.commons.cli.Option
import org.apache.commons.cli.Options
import java.util.*

internal class BiomedInteractor private constructor(
	private val Flags: String
) : Interactor
{
	private val Reader = Scanner( System.`in` )

	override fun getFlags(): String = this.Flags

	override fun getNextDocument(): String?
	{
		return	if( !this.Reader.hasNextLine() )	{ null }
				else								{ this.Reader.nextLine() }
	}

	override fun returnResult( Result: String ): Unit = println( Result )

	companion object Factory: InteractorFactory
	{
		private val ArgParser: CLI = CLIWrapper

		private fun setupOptions(): Options = this.ArgParser.getOptions()
												.addOption( this.buildFlagsOption() )

		private fun buildFlagsOption(): Option
		{
			return this.ArgParser.getOptionBuilder( "f" )
				.longOpt( "flags" )
				.desc( "Runs the program while skipping actually sending email, just log their content.")
				.required( true )
				.hasArg( true )
				.numberOfArgs( 1 )
				.type( String::class.java )
				.build()
		}

		private fun readArguments(
			Opts: Options,
			Arguments: Array<String>
		): CommandLine = this.ArgParser.getParser().parse( Opts, Arguments )

		private fun parseArgs(
			Arguments: Array<String>
		): String = this.readArguments( this.setupOptions(), Arguments ).getOptionValue( "flags" )

		override fun getInstance(
			Arguments: Array<String>
		): Interactor = BiomedInteractor( Flags = this.parseArgs( Arguments ) )
	}
}
