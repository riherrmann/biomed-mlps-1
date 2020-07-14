package de.huberlin.biomed

import de.huberlin.biomed.io.BiomedInteractor

fun main( args: Array<String> )
{
	BiomedComplex.getInstance(
		BiomedInteractor.getInstance( args )
	).run()
}