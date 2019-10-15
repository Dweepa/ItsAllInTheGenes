#!/bin/bash

depths=(1 2 3 4)
k=(1 2 3 4)
embedding_lengths=(8 16 32)
epochs=(50 75 100)


d_len=$((${#depths[@]}-1))
emb_len=$((${#embedding_lengths[@]}-1))
epoch_len=$((${#epochs[@]}-1))

for i in $(seq 0 $d_len)
do
	for j in $(seq 0 $emb_len)
	do
		python3 SNN.py ${depths[$i]} ${k[$i]} ${embedding_lengths[$j]}

		for k in $(seq 0 $epoch_len)
		do
			python save_embeddings.py ${depths[$i]} ${k[$i]} ${embedding_lengths[$j]} ${epochs[$k]}
			python internal_evaluation.py EMB_snn_${depths[$i]}_${k[$i]}_${embedding_lengths[$j]}-${epochs[$k]}
		done
	done
done