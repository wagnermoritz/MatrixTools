import torch
from typing import Final

class StreamGramMatrix:
    """
    Streaming computation of XX^T with Kahan summation.
    """

    __slots__ = ["d", "device", "output_dtype", "internal_dtype", "gram", "kahan_compensation"]

    def __init__(self, d: int, device: torch.device, output_dtype: torch.dtype, internal_dtype: torch.dtype = torch.double):
        """
        Initialize the StreamGramMatrix.

        Args:
            d: The dimension of the data
            device: The device to use
            output_dtype: The dtype of the output Gram matrix
            internal_dtype: The dtype of the internal Gram matrix and compensation term
        """
        self.d: Final[int] = d
        self.device = device
        self.output_dtype = output_dtype
        self.internal_dtype: Final[torch.dtype] = internal_dtype

        self.gram = torch.zeros(d, d, device=device, dtype=internal_dtype)
        self.kahan_compensation = torch.zeros(d, d, device=device, dtype=internal_dtype)

    def __repr__(self) -> str:
        return f"StreamGramMatrix(\n{self._get_repr_str()}\n)"
    
    def _get_repr_str(self) -> str:
        return f"  gram: Tensor of shape {list(self.gram.shape)}\n" \
               f"  kahan_compensation: Tensor of shape {list(self.kahan_compensation.shape)}\n" \
               f"  device: {self.device}\n" \
               f"  output_dtype: {self.output_dtype}\n" \
               f"  internal_dtype: {self.internal_dtype}\n" \
               f"  d: {self.d}"

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> "StreamGramMatrix":
        """
        Move the StreamGramMatrix to a different device and dtype.
        """
        if device is not None:
            self.device = device
            self.gram = self.gram.to(device=device)
            self.kahan_compensation = self.kahan_compensation.to(device=device)
        if dtype is not None:
            self.output_dtype = dtype
        return self

    def _check_tensor(self, tensor: torch.Tensor) -> None:
        """
        Check if argument is a tensor on the correct device and dtype.

        Args:
            tensor: The tensor to check
        """
        assert torch.is_tensor(tensor), "Argument is not a tensor"
        assert tensor.device == self.device, f"Tensor is not on the correct device ({tensor.device} != {self.device})"

    def get_gram_matrix(self) -> torch.Tensor:
        """
        Get the Gram matrix in the output dtype.
        """
        return self.gram.to(dtype=self.output_dtype)
    
    def add(self, X: torch.Tensor):
        """
        Add a batch of data to the Gram matrix.

        Args:
            X: The batch of data
        """
        self._check_tensor(X)
        assert len(X.shape) in [1, 2], "Tensor must be 1D or 2D"
        if X.dtype != self.internal_dtype:
            X = X.to(dtype=self.internal_dtype)
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        assert X.shape[1] == self.d, f"Tensor has wrong shape ({X.shape[1]} != {self.d})"
        
        self._kahan_add(X.T @ X)

    def _kahan_add(self, value: torch.Tensor):
        """
        Kahan summation for maximum numerical precision.

        Args:
            value: The value to add
        """
        y = value - self.kahan_compensation
        t = self.gram + y
        self.kahan_compensation = (t - self.gram) - y
        self.gram = t


class StreamGramMatrixMean(StreamGramMatrix):
    """
    Streaming computation of XX^T using the running mean and Kahan summation.
    """

    __slots__ = ["n_samples"] + StreamGramMatrix.__slots__
    
    def __init__(self, d: int, device: torch.device, output_dtype: torch.dtype, internal_dtype: torch.dtype = torch.double):
        super().__init__(d, device, output_dtype, internal_dtype)
        self.n_samples = 0

    def __repr__(self) -> str:
        return f"StreamGramMatrixMean(\n{self._get_repr_str()}\n)"
    
    def _get_repr_str(self) -> str:
        string = super()._get_repr_str()
        string += f"\n  n_samples: {self.n_samples}"
        return string

    def get_gram_matrix(self) -> torch.Tensor:
        """
        Get the Gram matrix in the output dtype.
        """
        return (self.gram * self.n_samples).to(dtype=self.output_dtype)
    
    def get_gram_matrix_mean(self) -> torch.Tensor:
        """
        Get the Gram matrix scaled by the number of samples in the output dtype.
        """
        return self.gram.to(dtype=self.output_dtype)
    
    def add(self, X: torch.Tensor):
        """
        Add a batch of data to the Gram matrix.

        Args:
            X: The batch of data
        """
        self._check_tensor(X)
        assert len(X.shape) in [1, 2], "Tensor must be 1D or 2D"
        if X.dtype != self.internal_dtype:
            X = X.to(dtype=self.internal_dtype)
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        assert X.shape[1] == self.d, f"Tensor has wrong shape ({X.shape[1]} != {self.d})"
        batch_size = X.shape[0]

        n_new = self.n_samples + batch_size
        update = X.T @ X
        update.sub_(self.gram * batch_size).div_(n_new)
        self._kahan_add(update)

        self.n_samples = n_new