#! /bin/bash


for j in "." ;do
	nom_train="obj"$j"/train"
	nom_test="obj"$j"/test"
	mkdir -p $nom_train $nom_test
	nombre = $(ls | wc -l)
	for i in $j ; do
		if [ $i -lt 256 ]
		then 
			mv "obj"$j"__"$i".png" $nom_train
		else
			mv "obj"$j"__"$i".png" $nom_test
		fi
	done
done

