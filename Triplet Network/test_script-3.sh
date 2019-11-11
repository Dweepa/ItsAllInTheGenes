#!/bin/bash
# sys arguments: layer neuron emb_len dropout samples_per_pert
layer=(5)
neuron=(512 1024)
embedding_lengths=(16 32)
dropout=(0.5 0.9)
epochs=(100)
samples_per_pert=(100)


layer_len=$((${#layer[@]}-1))
neuron_len=$((${#neuron[@]}-1))
emb_len=$((${#embedding_lengths[@]}-1))
drop_len=$((${#dropout[@]}-1))
epoch_len=$((${#epochs[@]}-1))
sample_len=$((${#samples_per_pert[@]}-1))

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
		      echo ""
          python main.py ${layer[$lay]} ${neuron[$neu]} ${embedding_lengths[$emb]} ${dropout[$drop]} ${samples_per_pert[$sample]}

	        for ep in $(seq 0 $epoch_len)
          do
            echo ""
            echo "Saving Embeddings for MOD_${layer[$lay]}_${neuron[$neu]}_${embedding_lengths[$emb]}_${dropout[$drop]}_${samples_per_pert[$sample]}-${epochs[$ep]}"
            python save_embeddings.py ${layer[$lay]} ${neuron[$neu]} ${embedding_lengths[$emb]} ${dropout[$drop]} ${samples_per_pert[$sample]} 0 ${epochs[$ep]}
            echo ""
            echo "Performing Internal Evaluation for EMB_${layer[$lay]}_${neuron[$neu]}_${embedding_lengths[$emb]}_${dropout[$drop]}_${samples_per_pert[$sample]}-${epochs[$ep]}"
            python ../Internal\ Evaluation/internal_evaluation.py EMB_triplet_${layer[$lay]}_${neuron[$neu]}_${embedding_lengths[$emb]}_${dropout[$drop]}_${samples_per_pert[$sample]}-${epochs[$ep]}
          done
        done
      done
    done
  done
done