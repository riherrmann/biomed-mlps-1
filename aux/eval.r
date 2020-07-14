BaseDir <- normalizePath( "." ) 
BaseLine <- as.data.frame.matrix(
	read.table(
		file = paste(
					BaseDir,
					"/../training_data/train_75.tsv",
					sep = ""
		),
		sep = '\t',
		header = TRUE,
		quote = "",
		stringsAsFactors = FALSE
	)
)


print( BaseLine[ "is_cancer" ] )
