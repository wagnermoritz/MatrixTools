import torch
from typing import Final, Any

class Rank1UpdatableTopkSVD:

    __slots__ = [
        'device','dtype','X','n','m','k','oversample','r','tol', 'update_count',
        'U','S','Vh','subspace_iters','use_expanded_subspace','recompute_every',
        'use_rand_trunk_svd', 'rand_trunk_svd_iters'
    ]

    def __init__(self, X: torch.Tensor, k: int, oversample: int = 0, tol: float = 1e-9,
                 subspace_iters: int = 1, use_expanded_subspace: bool = True, recompute_every: int = 0,
                 use_rand_trunk_svd: bool = False, rand_trunk_svd_iters: int = 7) -> None:
        '''
        Initialize the UpdatableSVD.

        Args:
            X:  The matrix to store (X is updated in place, so clone if you want to keep the original matrix)
            k:  The number of singular values to keep
            oversample:  The number of additional singular values to keep for numerical stability
            tol:  The tolerance used to determine if the SVD is updated and if the bases are extended
            subspace_iters:  The number of subspace iterations to perform. 0 = no subspace iteration.
            use_expanded_subspace:  Whether to expand the subspace for the subspace iteration by the orthogonal
                components of the rank-1 update. (Only applies if k + oversample <= min(n, m) - 2)
            recompute_every:  Recompute the SVD of the matrix after the rank-1 update every recompute_every updates.
            use_rand_trunk_svd:  Initialize and recompute the SVD of the matrix using a randomized truncated SVD
                instead of torch.linalg.svd. Worse accuracy than torch.linalg.svd -> only useful for very large matrices.
            rand_trunk_svd_iters:  The number of power iterations to perform on the randomized truncated SVD.
        '''
        self.device, self.dtype = X.device, X.dtype
        self.X = X
        self.n: Final[int] = X.shape[0]
        self.m: Final[int] = X.shape[1]
        self.k: Final[int] = min(k, self.n, self.m)
        self.r: Final[int] = min(self.k + max(0, oversample), min(self.n, self.m))
        self.tol: Final[float] = tol
        self.subspace_iters: Final[int] = subspace_iters
        self.use_expanded_subspace: Final[bool] = use_expanded_subspace and self.r <= min(self.n, self.m) - 2
        self.recompute_every: Final[int] = recompute_every
        self.update_count = 0
        self.use_rand_trunk_svd: Final[bool] = use_rand_trunk_svd
        self.rand_trunk_svd_iters: Final[int] = rand_trunk_svd_iters
        assert self.rand_trunk_svd_iters > 0, "rand_trunk_svd_iters must be positive"
        self._init_topr()

    def __repr__(self) -> str:
        return f"Rank1UpdatableTopkSVD(\n" \
               f"  X: Tensor of shape {list(self.X.shape)}\n" \
               f"  U: Tensor of shape {list(self.U.shape)}\n" \
               f"  S: Tensor of shape {list(self.S.shape)}\n" \
               f"  Vh: Tensor of shape {list(self.Vh.shape)}\n" \
               f"  device: {self.device}\n" \
               f"  dtype: {self.dtype}\n" \
               f"  tol: {self.tol}\n" \
               f"  r: {self.r}\n" \
               f"  n: {self.n}\n" \
               f"  m: {self.m}\n" \
               f"  k: {self.k}\n" \
               f"  oversample: {self.oversample}\n" \
               f"  subspace_iters: {self.subspace_iters}\n" \
               f"  use_expanded_subspace: {self.use_expanded_subspace}\n" \
               f"  recompute_every: {self.recompute_every}\n" \
               f"  update_count: {self._update_count}\n" \
               f")"

    @torch.no_grad()
    def _init_topr(self) -> None:
        '''
        Initialize the top-r factors of the SVD of the matrix.
        '''
        if self.use_rand_trunk_svd:
            self._rand_trunk_svd()
        else:
            U, S, Vh = torch.linalg.svd(self.X, full_matrices=False)
            self.U = U[:, :self.r].contiguous()
            self.S = S[:self.r].contiguous()
            self.Vh = Vh[:self.r, :].contiguous()

    def to(self, device=None, dtype=None) -> 'Rank1UpdatableTopkSVD':
        '''
        Move the UpdatableSVD to a different device and/or dtype.
        '''
        if device is not None: self.device = device
        if dtype is not None: self.dtype = dtype
        self.X  = self.X.to(self.device, self.dtype)
        self.U  = self.U.to(self.device, self.dtype)
        self.S  = self.S.to(self.device, self.dtype)
        self.Vh = self.Vh.to(self.device, self.dtype)
        return self

    def recompute_svd(self) -> None:
        '''
        Recompute the top-r factors of the SVD of the matrix.
        '''
        self._init_topr()

    def reconstruct(self) -> torch.Tensor:
        '''
        Reconstruct the matrix from the top-k factors of the SVD.
        '''
        return self.U[:, :self.k] @ torch.diag(self.S[:self.k]) @ self.Vh[:self.k, :]

    def rank1_update(self, u: torch.Tensor, v: torch.Tensor) -> None:
        '''
        Perform a rank-1 update to the matrix.
        '''
        self._check_vec(u, self.n); self._check_vec(v, self.m)
        self.X.add_(torch.outer(u, v))
        self._rank1_update_core(u, v)

    def replace_row(self, i: int, new_row: torch.Tensor) -> None:
        '''
        Replace the i-th row of the matrix with the new row.
        '''
        self._check_vec(new_row, self.m)
        delta = new_row - self.X[i, :]
        if delta.norm() < self.tol:
            self.X[i, :] = new_row
            return
        u = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        u[i] = 1.0
        self.X[i, :] = new_row
        self._rank1_update_core(u, delta)

    def replace_column(self, j: int, new_col: torch.Tensor) -> None:
        '''
        Replace the j-th column of the matrix with the new column.
        '''
        self._check_vec(new_col, self.n)
        delta = new_col - self.X[:, j]
        if delta.norm() < self.tol:
            self.X[:, j] = new_col
            return
        v = torch.zeros(self.m, dtype=self.dtype, device=self.device)
        v[j] = 1.0
        self.X[:, j] = new_col
        self._rank1_update_core(delta, v)

    def add_to_row(self, i: int, delta: torch.Tensor) -> None:
        '''
        Add the delta to the i-th row of the matrix.
        '''
        self._check_vec(delta, self.m)
        if delta.norm() < self.tol:
            self.X[i, :] += delta
            return
        u = torch.zeros(self.n, dtype=self.dtype, device=self.device)
        u[i] = 1.0
        self.X[i, :] += delta
        self._rank1_update_core(u, delta)

    def add_to_column(self, j: int, delta: torch.Tensor) -> None:
        '''
        Add the delta to the j-th column of the matrix.
        '''
        self._check_vec(delta, self.n)
        if delta.norm() < self.tol:
            self.X[:, j] += delta
            return
        v = torch.zeros(self.m, dtype=self.dtype, device=self.device)
        v[j] = 1.0
        self.X[:, j] += delta
        self._rank1_update_core(delta, v)  

    def _check_vec(self, t: torch.Tensor, length: int) -> None:
        '''
        Check if the tensor is a vector of the correct length and device and dtype.
        '''
        assert torch.is_tensor(t)
        assert len(t.shape) == 1 and t.shape[0] == length
        assert t.device == self.device and t.dtype == self.dtype

    @torch.no_grad()
    def _rand_trunk_svd(self) -> None:
        '''
        Approximation of the truncated SVD of X, based on the prototype for
        randomized SVD in https://arxiv.org/pdf/0909.4061
        (Finding structure in randomness: Probabilistic algorithms for constructing
        approximate matrix decompositions, N. Halko, P. G. Martinsson, J. A. Tropp)
        '''
        if self.m < self.n:
            # randomized range finder
            Y = self.X @ torch.randn(self.m, self.r, dtype=self.dtype, device=self.device)
            Y, _ = torch.linalg.qr(Y, mode='reduced')
            # sharpen spectrum with power iterations
            for i in range(self.rand_trunk_svd_iters):
                end_with_qr = False
                Y = self.X @ (self.X.T @ Y)
                if i % 4 == 0: # reorthogonalize during long power iteration
                    end_with_qr = True
                    Y, _ = torch.linalg.qr(Y, mode='reduced')
            if not end_with_qr:
                Y, _ = torch.linalg.qr(Y, mode='reduced')
            # small SVD + lift
            Ub, S, Vh = torch.linalg.svd(Y.T @ self.X, full_matrices=False)
            self.U = (Y @ Ub)[:, :self.r].contiguous()
            self.S = S[:self.r].contiguous()
            self.Vh = Vh[:self.r, :].contiguous()
        else:
            # randomized range finder
            Y = self.X.T @ torch.randn(self.n, self.r, dtype=self.dtype, device=self.device)
            Y, _ = torch.linalg.qr(Y, mode='reduced')
            # sharpen spectrum with power iterations
            for i in range(self.rand_trunk_svd_iters):
                end_with_qr = False
                Y = self.X.T @ (self.X @ Y)
                if i % 4 == 0: # reorthogonalize during long power iteration
                    end_with_qr = True
                    Y, _ = torch.linalg.qr(Y, mode='reduced')
            if not end_with_qr:
                Y, _ = torch.linalg.qr(Y, mode='reduced')
            # small SVD + lift
            Ub, S, Vh = torch.linalg.svd(self.X @ Y, full_matrices=False)
            self.U = Ub[:, :self.r].contiguous()
            self.S = S[:self.r].contiguous()
            self.Vh = (Y @ Vh.T).T[:self.r, :].contiguous()
        
        self._subspace_iteration(iters=3)

    @torch.no_grad()
    def _rank1_update_core(self, u: torch.Tensor, v: torch.Tensor) -> None:
        '''
        Brand-like update of the top-r factors of the SVD after a rank-1 update.
        Sec. 3 in https://www.merl.com/publications/docs/TR2006-059.pdf
        (Fast low-rank modifications of the thin singular value decomposition, M. Brand)
        '''
        self.update_count += 1
        if self.recompute_every > 0 and self.update_count % self.recompute_every == 0:
            self.recompute_svd()
            return

        if torch.norm(u) * torch.norm(v) < self.tol:
            return

        U, S, Vh = self.U, self.S, self.Vh
        V = Vh.T
        r = S.shape[0]

        # projections onto current subspaces
        a = U.T @ u
        b = V.T @ v

        # orthogonal residuals
        p = u - U @ a
        alpha = torch.norm(p)
        if alpha > self.tol: p = p / alpha
        else: p, alpha = None, 0.0

        q = v - V @ b
        beta = torch.norm(q)
        if beta > self.tol: q = q / beta
        else: q, beta = None, 0.0

        # build small core
        ru = r + (1 if p is not None else 0)
        rv = r + (1 if q is not None else 0)
        K = torch.zeros((ru, rv), dtype=self.dtype, device=self.device)

        # small core matrix lifted by rank-1
        K[:r, :r] = torch.diag(S) + torch.outer(a, b)
        if p is not None: K[r, :r] = alpha * b
        if q is not None: K[:r, rv-1] = beta * a
        if p is not None and q is not None: K[r, rv-1] = alpha * beta

        # SVD of core matrix
        Uc, Sc, Vch = torch.linalg.svd(K, full_matrices=False)

        # updated factors, expand old factors if needed
        U_new = (U if p is None else torch.cat([U, p[:, None]], dim=1)) @ Uc
        V_new = (V if q is None else torch.cat([V, q[:, None]], dim=1)) @ Vch.T
        S_new = Sc

        # truncate back to fixed working rank r (keep largest r)
        idx = torch.argsort(S_new, descending=True)[:self.r]
        self.U = U_new[:, idx]
        self.S = S_new[idx]
        self.Vh = V_new[:, idx].T

        if self.subspace_iters > 0:
            if self.use_expanded_subspace:
                self._subspace_iteration_expanded(u, v)
            else:
                self._subspace_iteration(u, v)

    @torch.no_grad()
    def _subspace_iteration(self, *args: tuple[Any, ...], iters: int = 0) -> None:
        '''
        Perform a subspace iteration to refine the top-r factors of the SVD.
        Algorithm 4.4 in https://arxiv.org/pdf/0909.4061 + Rayleigh-Ritz lift
        (Finding structure in randomness: Probabilistic algorithms for constructing
        approximate matrix decompositions, N. Halko, P. G. Martinsson, J. A. Tropp)
        '''
        
        X = self.X
        # orthonormalize V and iterate
        V, _ = torch.linalg.qr(self.Vh.T, mode='reduced')
        for _ in range(max(iters, self.subspace_iters)):
            Y = X @ V
            U, _ = torch.linalg.qr(Y, mode='reduced')
            Y = X.T @ U
            V, _ = torch.linalg.qr(Y, mode='reduced')
        Y = X @ V
        U, _ = torch.linalg.qr(Y, mode='reduced')

        # Rayleigh-Ritz
        B = U.T @ (X @ V)
        Ub, S, Vbh = torch.linalg.svd(B, full_matrices=False)
        self.U = U @ Ub
        self.S = S
        self.Vh = (V @ Vbh.T).T

    @torch.no_grad()
    def _subspace_iteration_expanded(self, u: torch.Tensor, v: torch.Tensor) -> None:
        '''
        Perform a subspace iteration to refine the top-r factors of the SVD.
        Expands the subspace by the orthogonal components of the rank-1 update.
        Algorithm 4.4 in https://arxiv.org/pdf/0909.4061 + Rayleigh-Ritz lift + basis extension
        (Finding structure in randomness: Probabilistic algorithms for constructing
        approximate matrix decompositions, N. Halko, P. G. Martinsson, J. A. Tropp)
        '''
        X = self.X
        # expand V by orthogonal components of rank-1 update
        v_orth = v - self.Vh.T @ self.Vh @ v
        V = torch.cat([self.Vh.T, v_orth[:, None]], dim=1)
        y = X.T @ u - torch.dot(u, u) * v
        y_orth = y - V @ V.T @ y
        V = torch.cat([V, y_orth[:, None]], dim=1)
        # orthonormalize V and iterate
        V, _ = torch.linalg.qr(V, mode='reduced')
        for _ in range(self.subspace_iters):
            Y = X @ V
            U, _ = torch.linalg.qr(Y, mode='reduced')
            Y = X.T @ U
            V, _ = torch.linalg.qr(Y, mode='reduced')
        Y = X @ V
        U, _ = torch.linalg.qr(Y, mode='reduced')

        # Rayleigh-Ritz and truncate
        B = U.T @ (X @ V)
        Ub, S, Vbh = torch.linalg.svd(B, full_matrices=False)
        self.U = (U @ Ub)[:, :self.r].contiguous()
        self.S = S[:self.r].contiguous()
        self.Vh = (V @ Vbh.T)[:, :self.r].T.contiguous()