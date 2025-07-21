import os
import numpy as np
from ts2vec.ts2vec import TS2Vec
import torch
from timeseries_data import load_data_sc
import time

sc_path = "/gpfs01/berens/data/data/superior_colliculus/"
result_path = "/gpfs01/berens/user/sdamrich/data/cl_neuro/ts2vec"


# first hyperparameter set is the default, second is adapted to our setup
# hyperparameters are n_epochs, batch_size, similarity type, and embedding dimension
other_hparams = [
    [None, 16, "cosine", 320], # TS2Vec+PCA
    [1000, 768, "cauchy", 2] # TRACE+TS2Vec
]

seed = 2

for n_epochs, batch_size, similarity, output_dims in other_hparams:
    print("loading data")
    data = load_data_sc(flatten_bar=False)

    # compute mean trial responses
    mean_data = np.concatenate([np.mean(d, axis=1) for d in data[:2]], axis=-1)

    # set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Train a TS2Vec model
    model = TS2Vec(
        input_dims=1,
        device=0,
        output_dims=output_dims,
        batch_size=batch_size

    )
    print("starting training")
    start = time.time()
    loss_log = model.fit(
        mean_data[:, :, None],
        verbose=True,
        n_epochs=n_epochs,
        similarity=similarity,
    )

    embd = model.encode(mean_data[:, :, None], encoding_window='full_series')
    end = time.time()
    print(f"done after {end-start} seconds")
    file_name = f"sc_dat_ts2vec_epochs_{n_epochs}_bs_{batch_size}_sim_{similarity}_dim_{output_dims}_seed_{seed}_embd.npz"
    np.savez(os.path.join(result_path, file_name), embd=embd, time=end-start)
    print(f"Saved {file_name}")
