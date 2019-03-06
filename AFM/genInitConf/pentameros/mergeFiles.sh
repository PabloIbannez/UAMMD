model=61
for f in *.pdb; do
	echo "MODEL        "$model >> merged
	cat $f >> merged
	echo "ENDMDL" >> merged
	let "model=model+1"
done
