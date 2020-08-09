package de.huberlin.biomed.io

import org.apache.commons.cli.CommandLineParser
import org.apache.commons.cli.HelpFormatter
import org.apache.commons.cli.Option
import org.apache.commons.cli.Options
import org.junit.Test
import org.junit.runner.RunWith
import org.powermock.modules.junit4.PowerMockRunner
import kotlin.test.assertTrue

@RunWith( PowerMockRunner::class )
class CLIWrapperSpec
{
	@Test
	fun `it creates a Options object`()
	{
		val MyOption: Any = CLIWrapper.getOptions()
		assertTrue( MyOption is Options )
	}

	@Test
	fun `it creates a OptionBuilder object`()
	{
		val MyOption: Any = CLIWrapper.getOptionBuilder()
		assertTrue( MyOption is Option.Builder )
	}

	@Test
	fun `it creates a named OptionBuilder object`()
	{
		val MyOption: Any = CLIWrapper.getOptionBuilder( "f" )
		assertTrue( MyOption is Option.Builder )
	}

	@Test
	fun `it creates a CMDParser object`()
	{
		val MyOption: Any = CLIWrapper.getParser()
		assertTrue( MyOption is CommandLineParser )
	}

	@Test
	fun `it creates a HelpFormatter object`()
	{
		val MyOption: Any = CLIWrapper.getHelpFormatter()
		assertTrue( MyOption is HelpFormatter )
	}
}