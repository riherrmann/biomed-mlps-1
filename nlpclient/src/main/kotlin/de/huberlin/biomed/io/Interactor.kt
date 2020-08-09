package de.huberlin.biomed.io

interface Interactor
{
	fun getFlags(): String
	fun getNextDocument(): String?
	fun returnResult( Result: String )
}

interface InteractorFactory
{
	fun getInstance( Arguments: Array<String> ): Interactor
}