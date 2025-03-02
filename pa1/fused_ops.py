from typing import Any, Dict, List
import torch
from auto_diff import *


class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""
    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape over which to compute mean and variance (typically the last few dims).
            eps: A small constant to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        A, B = input_values
        X = torch.matmul(A, B)
        normalized_shape = node.attrs["normalized_shape"]
        dims = tuple(range(-len(normalized_shape), 0))
        mu = X.mean(dim=dims, keepdim=True)
        diff = X - mu
        var = (diff * diff).mean(dim=dims, keepdim=True)
        std = torch.sqrt(var + node.attrs["eps"])
        y = diff / std
        node.attrs["mu"] = mu
        node.attrs["std"] = std
        node.attrs["diff"] = diff
        node.attrs["dims"] = dims
        node.attrs["X"] = X
        return y

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        A = node.inputs[0]
        B = node.inputs[1]
        normalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]
        dims = node.attrs.get("dims", tuple(range(-len(normalized_shape), 0)))

        X = matmul(A, B)
        mu = mean(X, dims, keepdim=True)
        diff = sub(X, mu)
        var = mean(mul(diff, diff), dims, keepdim=True)
        std = sqrt(add_by_const(var, eps))
        y = div(diff, std)

        mean_output_grad = mean(output_grad, dims, keepdim=True)
        output_grad_times_y = mul(output_grad, y)
        mean_output_grad_times_y = mean(output_grad_times_y, dims, keepdim=True)
        numerator = sub(sub(output_grad, mean_output_grad), mul(y, mean_output_grad_times_y))
        dX = div(numerator, std)

        dA = matmul(dX, transpose(B, -2, -1))
        dB = matmul(transpose(A, -2, -1), dX)
        return [dA, dB]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""
    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        A, B = input_values
        X = torch.matmul(A, B)
        dim = node.attrs["dim"]
        y = torch.softmax(X, dim=dim)
        node.attrs["y"] = y
        node.attrs["X"] = X
        return y

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints for A and B."""
        A = node.inputs[0]
        B = node.inputs[1]
        dim = node.attrs["dim"]
        X = matmul(A, B)
        y = softmax(X, dim=dim)
        
        sg = mul(y, output_grad)
        sum_sg = sum_op(sg, dim=dim, keepdim=True)
        dX = mul(y, sub(output_grad, sum_sg))

        dA = matmul(dX, transpose(B, -2, -1))
        dB = matmul(transpose(A, -2, -1), dX)
        return [dA, dB]


matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()
