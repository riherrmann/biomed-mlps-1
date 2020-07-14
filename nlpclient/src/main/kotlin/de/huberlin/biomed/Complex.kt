package de.huberlin.biomed

import de.huberlin.biomed.io.Interactor

interface Complex
{
	fun run()
}

interface ComplexFactory
{
	fun getInstance( Interaction: Interactor ): Complex
}