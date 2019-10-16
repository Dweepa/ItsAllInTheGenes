#!/bin/bash

depths=(4 8 16 24 32 4 8 16 24 32 4 8 16 24 32 4 8 16 24 32)
k_val=(25 12 6 4 3 48 24 12 8 6 90 44 22 15 11 128 63 31 21 15)
embedding_lengths=(8 16 32)
epochs=(50 75 100)


d_len=$((${#depths[@]}-1))
emb_len=$((${#embedding_lengths[@]}-1))
epoch_len=$((${#epochs[@]}-1))

for i in $(seq 0 $d_len)
do
	for j in $(seq 0 $emb_len)
	do
		echo ""
		echo "Creating Models for MOD_${depths[$i]}_${k_val[$i]}_${embedding_lengths[$j]}"
		python3 SNN.py ${depths[$i]} ${k_val[$i]} ${embedding_lengths[$j]} 100 25

		for k in $(seq 0 $epoch_len)
		do
			echo ""
			echo "Saving Embeddings for MOD_${depths[$i]}_${k_val[$i]}_${embedding_lengths[$j]}-${epochs[$k]}"
			python3 save_embeddings.py ${depths[$i]} ${k_val[$i]} ${embedding_lengths[$j]} 0 ${epochs[$k]}
			echo ""
			echo "Performing Internal Evaluation for EMB_${depths[$i]}_${k_val[$i]}_${embedding_lengths[$j]}-${epochs[$k]}"
			python3 ../Internal\ Evaluation/internal_evaluation.py EMB_snn_${depths[$i]}_${k_val[$i]}_${embedding_lengths[$j]}-${epochs[$k]}
		done
	done
done