#/bin/bash

# file name standarization
# remove spaces and change upper case to lower case
# move them to a new directory
# push them to pachyderm

INPUT_DIR="./audio-unprocessed"
OUTPUT_DIR="./audio-processed"

renamefiles () {
	for oldname in "$INPUT_DIR"/*.mp3; do 
		newname=`echo $oldname | sed 's/ //g' | tr '[:upper:]' '[:lower:]'` 
		mv "$oldname" "$newname"
	done
}

movefiles () {
	moveme=`mv $INPUT_DIR/*.mp3 $OUTPUT_DIR/`
	echo "$moveme"
}

pachyput () {
	    cd $OUTPUT_DIR
		for file in ./*.mp3; do
			newfilename=`basename $file` 
			pachypush=`pachctl put file audio-unprocessed@master:$newfilename -f $file`
			echo "$pachypush"
		done
}

renamefiles
movefiles
pachyput