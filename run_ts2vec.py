import os
import numpy as np
from ts2vec.ts2vec import TS2Vec
import torch
import time


toy_path = "/gpfs01/berens/user/sdamrich/data/cl_neuro/toy_data"
result_path = "/gpfs01/berens/user/sdamrich/data/cl_neuro/ts2vec"

# std dev of the baseline noise
basenoise_SDs = [1, 2, 4, 6, 10, 15, 20, 24, 38, 60]
# first hyperparameter set is the default, second is adapted to our setup
# hyperparameters are n_epochs, batch_size, similarity type, and embedding dimension
other_hparams = [
    [None, 16, "cosine", 320], # TS2Vec + PCA (without the PCA step)
    [1000, 768, "cauchy", 2], # TRACE + TS2Vec
]

seeds = [0, 1, 2]

for n_epochs, batch_size, similarity, output_dims in other_hparams:
    for basenoise_SD in basenoise_SDs:
        for seed in seeds:
            file_name = f"toy_data_signalampl1_basenoiseSD{basenoise_SD}_signalnoiseSD8.npy"

            data = np.load(os.path.join(toy_path, file_name))
            mean_data = np.mean(data, axis=1)

            np.random.seed(seed)
            torch.manual_seed(seed)

            # Train a TS2Vec model
            model = TS2Vec(
                input_dims=1,
                device=0,
                output_dims=output_dims,
                batch_size=batch_size

            )
            start = time.time()
            loss_log = model.fit(
                mean_data[:, :, None],
                verbose=True,
                n_epochs=n_epochs,
                similarity=similarity,
            )
            embd = model.encode(mean_data[:, :, None], encoding_window='full_series')
            end = time.time()
            file_name = f"toy_data_basenoiseSD_{basenoise_SD}_ts2vec_epochs_{n_epochs}_bs_{batch_size}_sim_{similarity}_dim_{output_dims}_seed_{seed}_embd.npz"
            np.savez(os.path.join(result_path, file_name), embd=embd, time=end-start)
            print(f"Saved {file_name}")
