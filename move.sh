#! /bin/bash


for j in $(seq 1 100);do
	nom_train="obj"$j"/train"
	nom_test="obj"$j"/test"
	mkdir -p $nom_train $nom_test
	for i in $(seq 0 355); do
		if [ $i -lt 256 ]
		then 
			mv "obj"$j"__"$i".png" $nom_train
		else
			mv "obj"$j"__"$i".png" $nom_test
		fi
	done
done

