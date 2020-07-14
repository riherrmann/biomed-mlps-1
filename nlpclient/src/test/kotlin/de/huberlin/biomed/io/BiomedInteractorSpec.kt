package de.huberlin.biomed.io

import org.apache.commons.cli.CommandLine
import org.apache.commons.cli.CommandLineParser
import org.apache.commons.cli.Option
import org.apache.commons.cli.Options
import org.junit.After
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.ArgumentMatchers.anyString
import org.mockito.Mock
import org.mockito.Mockito.*
import org.powermock.api.mockito.PowerMockito
import org.powermock.api.mockito.PowerMockito.mockStatic
import org.powermock.api.mockito.PowerMockito.whenNew
import org.powermock.core.classloader.annotations.PrepareForTest
import org.powermock.modules.junit4.PowerMockRunner
import org.powermock.reflect.Whitebox
import java.io.ByteArrayOutputStream
import java.io.Console
import java.io.PrintStream
import java.io.PrintWriter
import java.util.*
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

@PrepareForTest(
	BiomedInteractor::class,
	Option::class,
	PrintWriter::class,
	Scanner::class,
	Console::class
)
@RunWith( PowerMockRunner::class )
class BiomedInteractorSpec
{
	private val Stdout = System.out
	private lateinit var Reader: Scanner
	private lateinit var OutputSpy: ByteArrayOutputStream

	private lateinit var Wrapper: CLI
	private lateinit var Args: Options
	private lateinit var FlagsBuilder: Option.Builder
	private lateinit var Flags: Option
	private lateinit var Parser: CommandLineParser
	private lateinit var ParsedArgs: CommandLine

	private val CMDArguments = arrayOf(
		"-f",
		"na"
	)

	@Before
	fun setup()
	{
		this.Args = mock( Options::class.java )
		this.Wrapper = mock( CLI::class.java )
		this.FlagsBuilder = mock( Option.Builder::class.java )
		this.Flags = mock( Option::class.java )
		this.Parser = mock( CommandLineParser::class.java )
		this.ParsedArgs = mock( CommandLine::class.java )

		this.Reader = PowerMockito.mock( Scanner::class.java )

		this.OutputSpy = ByteArrayOutputStream()
		System.setOut( PrintStream( this.OutputSpy ) )

		mockStatic( Option::class.java )
		whenNew( Scanner::class.java ).withAnyArguments().thenReturn( this.Reader )

		`when`( this.Wrapper.getOptions() ).thenReturn( this.Args )
		`when`( this.Wrapper.getOptionBuilder( anyString() ) ).thenReturn( this.FlagsBuilder )
		`when`( this.Args.addOption( this.Flags ) ).thenReturn( this.Args )
		`when`( this.Wrapper.getParser() ).thenReturn( this.Parser )

		`when`( this.FlagsBuilder.longOpt( anyString()) ).thenReturn( this.FlagsBuilder )
		`when`( this.FlagsBuilder.desc( anyString() ) ).thenReturn( this.FlagsBuilder )
		`when`( this.FlagsBuilder.required( true ) ).thenReturn( this.FlagsBuilder )
		`when`( this.FlagsBuilder.type( String::class.java ) ).thenReturn( this.FlagsBuilder )
		`when`( this.FlagsBuilder.hasArg( true ) ).thenReturn( this.FlagsBuilder )
		`when`( this.FlagsBuilder.numberOfArgs( 1 ) ).thenReturn( this.FlagsBuilder )
		`when`( this.FlagsBuilder.build() ).thenReturn( this.Flags )
		`when`( this.Parser.parse( this.Args, this.CMDArguments ) ).thenReturn( this.ParsedArgs )
		`when`( this.ParsedArgs.getOptionValue( anyString() ) ).thenReturn( "something" )

		`when`( this.Reader.nextLine() ).thenReturn("something" )
		`when`( this.Reader.hasNextLine() ).thenReturn( true )

		Whitebox.setInternalState( BiomedInteractor::class.java, "ArgParser", this.Wrapper )
	}

	@After
	fun tearDown()
	{
		System.setOut( this.Stdout )
		Whitebox.setInternalState( BiomedInteractor::class.java, "ArgParser", CLIWrapper )
	}

	@Test
	fun `it is a Interactor`()
	{
		val MyInteractions: Any = BiomedInteractor.getInstance( this.CMDArguments )
		assertTrue( MyInteractions is Interactor )
	}

	@Test
	fun it_fails_if_no_flags_had_been_given()
	{
		val Args = arrayOf<String>()
		val ExpectedException = Exception( "some errors" )
		`when`( this.Parser.parse( this.Args, Args ) )
			.then { throw ExpectedException }
			.thenReturn( this.ParsedArgs )

		val Exception = assertFailsWith( Exception::class )
		{ BiomedInteractor.getInstance ( Args ) }

		assertEquals(
			actual = Exception,
			expected = ExpectedException
		)
	}

	@Test
	fun `it returns the given flags`()
	{
		`when`( this.ParsedArgs.getOptionValue( "flags" ) ).thenReturn( this.CMDArguments[ 1 ] )

		val MyInteractor = BiomedInteractor.getInstance ( this.CMDArguments )
		assertEquals(
			actual = MyInteractor.getFlags(),
			expected = this.CMDArguments[ 1 ]
		)
	}

	@Test
	fun `it reads a line from stdin`()
	{
		val Expected = "My Doc string"

		this.Reader = PowerMockito.mock( Scanner::class.java )
		whenNew( Scanner::class.java ).withAnyArguments().thenReturn( this.Reader )

		`when`( this.Reader.nextLine() ).thenReturn( Expected )
		`when`( this.Reader.hasNextLine() ).thenReturn( true )

		val MyInteractor = BiomedInteractor.getInstance ( this.CMDArguments )
		assertEquals(
			actual = MyInteractor.getNextDocument(),
			expected = Expected
		)
	}

	@Test
	fun `it returns null if no input is present`()
	{
		val Expected = "My Doc string"
		this.Reader = PowerMockito.mock( Scanner::class.java )
		whenNew( Scanner::class.java ).withAnyArguments().thenReturn( this.Reader )

		`when`( this.Reader.nextLine() ).thenReturn( Expected )
		`when`( this.Reader.hasNextLine() ).thenReturn( false )

		val MyInteractor = BiomedInteractor.getInstance ( this.CMDArguments )
		assertNull( MyInteractor.getNextDocument() )
	}

	@Test
	fun `it prints to stdout`()
	{
		val ExpectedMessage = "result"
		val MyInteractor = BiomedInteractor.getInstance ( this.CMDArguments )
		MyInteractor.returnResult( ExpectedMessage )
		assertEquals(
			expected = ExpectedMessage + "\n",
			actual = this.OutputSpy.toString()
		)
	}
}