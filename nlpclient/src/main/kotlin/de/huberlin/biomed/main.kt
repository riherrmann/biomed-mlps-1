package de.huberlin.biomed

import de.huberlin.biomed.io.BiomedInteractor

fun main( args: Array<String> )
{
	try
	{
		BiomedComplex.getInstance(
			BiomedInteractor.getInstance( args )
		).run()
	} catch ( E: Exception )	{ System.err.println( E.message ) }
}