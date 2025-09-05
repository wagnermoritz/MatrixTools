# Tools for working with big matrices in PyTorch
In this repo, I collect implementations of algorithms for efficient, numerically stable, and memory-conscious processing of large-scale matrix operations. The file `test.ipynb` contains some small-scale example usage and error plots.

## Gram matrix streamer
The `gram_matrix_streamer.py` file contains two implementations of Kahan's summation algorithm for building the Gram matrix of large data sets that don't fit in memory. One implementation computes a Knuth-style runnig mean so the magnitudes of the entries do not overflow.

In this application "Gram matrix" referres to the _feature_ Gram matrix $X^\top X$ whose size depends on the number of dimensions, not the number of data points.

**Key features of `StreamGramMatrix`:**
- **Kahan summation**: Maintains a compensation term to reduce numerical errors
- **Memory efficient**: Processes data in batches without storing the full dataset
- **dtype flexibility**: Supports different internal and output precisions

**Additional features of `StreamGramMatrixMean`:**
- **Running mean**: Computes scaled Gram matrix using incremental mean updates
- **Overflow protection**: Prevents numerical overflow by maintaining normalized magnitudes
- **Dual output modes**: Can return both the full Gram matrix and the mean-scaled version

**Usage:**
```python
import torch
from gram_matrix_streamer import StreamGramMatrix

# initialize for 100-dimensional data
gram_computer = StreamGramMatrix(
    d=100, 
    device=torch.device('cuda'), 
    output_dtype=torch.float32,
    internal_dtype=torch.double  # higher precision for computations
)

# add data in batches
for batch in data_loader:
    gram_computer.add(batch)  # batch shape: (batch_size, 100)

# get the final Gram matrix X.T @ X in float32
gram_matrix = gram_computer.get_gram_matrix()
```

- Kahan, W. (1965). Further remarks on reducing truncation errors, commun. Assoc. Comput. Mach, 8, 40.
- Donald, E. K. (1999). The art of computer programming. Sorting and searching, 3(426-458), 4.

## Rank-1 updater for truncated SVD
The `rank1_topk_SVD_updater.py` file contains an implementation of Brand`s rank-1 SVD update with optional refinement through subspace iteration.

**Key features:**
- **Brand's algorithm**: Efficient rank-1 SVD updates without full recomputation
- **Truncated SVD with oversampling**: Maintains the top-(k + oversample) singular values and vectors for better approximation of the top-k SVD
- **Subspace iteration refinement**: Optional iterative refinement for improved accuracy
- **Subspace expansion**: Optionally expand the subspace for subspace iteration by the orthogonal components introduced by the rank-1 update
- **Multiple update types**: Supports various matrix modifications (rank-1 updates, row/column replacements/updates)
- **Periodic recomputation**: Includes optional periodic recomputation of the full SVD for better stability
- **Randomized truncated SVD**: Option to use randomized truncated SVD instead of `torch.linalg.svd` for very large matrices

**Usage:**
```python
import torch
from rank1_topk_SVD_updater import Rank1UpdatableTopkSVD

# initialize with a matrix
X = torch.randn(1024, 512, device=torch.device('cpu'))
svd_updater = Rank1UpdatableTopkSVD(
    X=X.clone(),                # clone to preserve original
    k=64,                       # keep top 64 singular values
    oversample=8,               # Uuse 8 extra for stability
    subspace_iters=2,           # 2 refinement iterations
    use_expanded_subspace=True, # expand the subspace for a better warm
                                #     start to the subspace iteration
    recompute_every=0,          # never recompute the SVD from scratch
    use_rand_trunk_svd=False    # matrix small enough for full SVD initialization
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# move everything to the GPU if available
svd_updater.to(device=device)

# general rank-1 update
u = torch.randn(1024, device=device)
v = torch.randn(512, device=device)
svd_updater.rank1_update(u, v)  # add outer(u, v) to matrix

# replace a row
new_row = torch.randn(512, device=device)
row_idx = 10
svd_updater.replace_row(row_idx, new_row)

# add to a column
delta_col = torch.randn(1024, device=device)
col_idx = 25
svd_updater.add_to_column(col_idx, delta_col)

# reconstruct the matrix from top-k factors
X_approx = svd_updater.reconstruct()

# access the top-(k + oversample) SVD factors directly
U = svd_updater.U
S = svd_updater.S
Vh = svd_updater.Vh
```

- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review, 53(2), 217-288.
- Brand, M. (2006). Fast low-rank modifications of the thin singular value decomposition. Linear algebra and its applications, 415(1), 20-30.