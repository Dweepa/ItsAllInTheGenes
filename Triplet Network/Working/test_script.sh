#!/bin/bash
# sys arguments: layer neuron emb_len dropout samples_per_pert
layer=(3 4 5 10 20)
neuron=(8 16 32 64 128 256 512 1024)
embedding_lengths=(8 16 32)
dropout=(0 0.1 0.2 0.5 0.8 0.9)
epochs=(50 75 100 200)
samples_per_pert=(50 100 200)


layer_len=$((${#layer[@]}-1))
neuron_len=$((${#neuron[@]}-1))
emb_len=$((${#embedding_lengths[@]}-1))
drop_len=$((${#dropout[@]}-1))
epoch_len=$((${#epochs[@]}-1))
sample_len=$((${#samples_per_pert[@]}-1))
echo ${sample_len}

for lay in $(seq 0 $layer_len)
do
	for neu in $(seq 0 $neuron_len)
	do
	  for emb in $(seq 0 $emb_len)
	  do
	    for drop in $(seq 0 $drop_len)
	    do
	      for sample in $(seq 0 $sample_len)
	      do
	        echo ""
		      echo "Creating Models for MOD_${layer[$lay]}_${neuron[$neu]}_${embedding_lengths[$emb]}_${dropout[$drop]}_${samples_per_pert[$sample]}"

	        for ep in $(seq 0 $epoch_len)
	        do
	          echo ""
            echo "Saving Embeddings for MOD_${layer[$lay]}_${neuron[$neu]}_${embedding_lengths[$emb]}_${dropout[$drop]}_${samples_per_pert[$sample]}-${epochs[$k]}"
            python3 main.py ${layer[$lay]} ${neuron[$neu]} ${embedding_lengths[$emb]} ${dropout[$drop]} ${samples_per_pert[$sample]}
            echo ""
            echo "Performing Internal Evaluation for EMB_${layer[$lay]}_${neuron[$neu]}_${embedding_lengths[$emb]}_${dropout[$drop]}_${samples_per_pert[$sample]}"
            python3 ../../Internal\ Evaluation/internal_evaluation.py EMB_triplet_${depths[$i]}_${k_val[$i]}_${embedding_lengths[$j]}-${epochs[$k]}
          done

        done
      done
    done
  done
done