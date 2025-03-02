from typing import Any, Dict, List

import torch


class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = {}, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        if attr_name == "shape":
            return ()
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[torch.Tensor]
            The input values of the given node.

        Returns
        -------
        output: torch.Tensor
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and there values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return input_values[0] + input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] + node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of input values."""
        assert len(input_values) == 2
        return input_values[0] * input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        return [output_grad * node.inputs[1], output_grad * node.inputs[0]]


class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] * node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        return [output_grad * node.constant]
    
class GreaterThanOp(Op):
    """Op to compare if node_A > node_B element-wise."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}>{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return element-wise comparison result as float tensor."""
        assert len(input_values) == 2
        return (input_values[0] > input_values[1]).float()

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Comparison operations have gradient of 0."""
        return [zeros_like(node.inputs[0]), zeros_like(node.inputs[1])]

class NoOp(Op):
    def __call__(self, value: float, name: str = "") -> Node:
        return Node(inputs=[], op=self, attrs={"value": value}, name=name)
    
    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        return torch.tensor(node.attrs["value"], dtype=torch.float32)
    
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return []
    
class SubOp(Op):
    """Op to element-wise subtract two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}-{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise subtraction of input values."""
        assert len(input_values) == 2
        return input_values[0] - input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of subtraction node, return partial adjoint to each input."""
        return [output_grad, mul_by_const(output_grad, -1)]
    
class OnesLikeOp(Op):
    def __call__(self, node_A: Node) -> Node:
        new_node = Node(inputs=[node_A], op=self, name=f"OnesLike({node_A.name})")
        shape = node_A.attrs.get("shape", None)
        if shape is None:
            shape = ()  
        new_node.attrs["shape"] = shape
        return new_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.ones_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]




class ZerosLikeOp(Op):
    def __call__(self, node_A: Node) -> Node:
        new_node = Node(inputs=[node_A], op=self, name=f"ZerosLike({node_A.name})")
        shape = node_A.attrs.get("shape", None)
        if shape is None:
            shape = ()
        new_node.attrs["shape"] = shape
        return new_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.zeros_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]

class UnsqueezeOp(Op):
    def __call__(self, node: Node, dim: int) -> Node:
        new_node = Node([node], self, attrs={"dim": dim}, name=f"Unsqueeze({node.name}, {dim})")
        in_shape = node.attrs.get("shape", None)
        if in_shape is None:
            in_shape = ()
        new_shape = list(in_shape)
        if dim < 0:
            dim = len(new_shape) + 1 + dim
        new_shape.insert(dim, 1)
        new_node.attrs["shape"] = tuple(new_shape)
        return new_node

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        result = torch.unsqueeze(input_values[0], node.attrs["dim"])
        # Set shape based on the computed tensor.
        node.attrs["shape"] = result.shape
        return result

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [output_grad]

unsqueeze = UnsqueezeOp()


class SumOp(Op):
    def __call__(self, node_A: Node, dim, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Sum({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        x = input_values[0]
        result = x.sum(dim=node.attrs["dim"], keepdim=node.attrs["keepdim"])
        node.attrs["input_shape"] = list(x.shape)
        node.attrs["shape"] = list(result.shape)
        return result

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        # Retrieve the original input shape stored during the forward pass.
        x_shape = node.attrs.get("input_shape", None)
        if x_shape is None or not x_shape:
            raise ValueError("Input shape not set for SumOp; run forward pass first.")
        
        dims = node.attrs["dim"]
        if not isinstance(dims, (tuple, list)):
            dims = (dims,)
        keepdim = node.attrs["keepdim"]


        grad = output_grad


        if not keepdim:
            dims = sorted([d if d >= 0 else d + len(x_shape) for d in dims])
            for d in dims:
                grad = unsqueeze(grad, d)  # unsqueeze should update grad.attrs["shape"]


        grad = expand_as(grad, node.inputs[0])
        return [grad]



class ExpandAsOp(Op):
    """Op to broadcast a tensor to the shape of another tensor.
    
    Note: This is a reference implementation for ExpandAsOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        return input_tensor.expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""
        
        return [sum_op(output_grad,dim=0), zeros_like(output_grad)]
    
class ExpandAsOp3d(Op):
    """Op to broadcast a tensor to the shape of another tensor.
    
    Note: This is a reference implementation for ExpandAsOp3d.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        #print('expand_op',input_tensor.shape, target_tensor.shape)
        if input_tensor.dim() == 0:
            return input_tensor.unsqueeze(0).expand_as(target_tensor)
        else:
            return input_tensor.unsqueeze(1).expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""
        
        return [sum_op(output_grad,dim=(0, 1)), zeros_like(output_grad)]

class LogOp(Op):
    """Logarithm (natural log) operation."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Log({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the natural logarithm of the input."""
        assert len(input_values) == 1, "Log operation requires one input."
        return torch.log(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the Log node, return the partial adjoint to the input."""
        input_node = node.inputs[0]
        return [output_grad / input_node]


class BroadcastOp(Op):
    def __call__(self, node_A: Node, input_shape: List[int], target_shape: List[int]) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"input_shape": input_shape, "target_shape": target_shape},
            name=f"Broadcast({node_A.name}, {target_shape})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 1
        return input_values[0].expand(node.attrs["target_shape"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of broadcast node, return partial adjoint to input.
        
        For broadcasting, we need to sum out the broadcasted dimensions to get
        back to the original shape.
        """
        if "input_shape" not in node.attrs:
            raise ValueError("Input shape is not set. Make sure compute() is called before gradient()")
            
        input_shape = node.attrs["input_shape"]
        output_shape = node.attrs["target_shape"]
        
        dims_to_sum = []
        for i, (in_size, out_size) in enumerate(zip(input_shape[::-1], output_shape[::-1])):
            if in_size != out_size:
                dims_to_sum.append(len(output_shape) - 1 - i)
                
        grad = output_grad
        if dims_to_sum:
            grad = sum_op(grad, dim=dims_to_sum, keepdim=True)
            
        if len(output_shape) > len(input_shape):
            grad = sum_op(grad, dim=list(range(len(output_shape) - len(input_shape))), keepdim=False)
            
        return [grad]

class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of input values."""
        assert len(input_values) == 2
        return input_values[0] / input_values[1]
    

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        x = node.inputs[0]
        y = node.inputs[1]
        grad_x = div(output_grad, y)
        y_sq = power(y, 2)
        grad_y = mul(mul_by_const(output_grad, -1), div(x, y_sq))
        return [grad_x, grad_y]


class DivByConstOp(Op):
    
    """Op to element-wise divide a nodes by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0]/node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [output_grad / node.constant]

class TransposeOp(Op):
    """Op to transpose a matrix."""

    def __call__(self, node_A: Node, dim0: int, dim1: int) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim0": dim0, "dim1": dim1},
            name=f"transpose({node_A.name}, {dim0}, {dim1})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the transpose of the input by swapping two dimensions.
        
        For example:
        - transpose(x, 1, 0) swaps first two dimensions
        """
        assert len(input_values) == 1
        in_tensor = input_values[0]
        
        dim0 = node.attrs['dim0']
        dim1 = node.attrs['dim1']
        return in_tensor.transpose(dim0, dim1)
                

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        dim0 = node.attrs['dim0']
        dim1 = node.attrs['dim1']
        
        grad = transpose(output_grad, dim0, dim1)
        return [grad]
        """Given gradient of transpose node, return partial adjoint to input."""
        """TODO: your code here"""

class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(
        self, node_A: Node, node_B: Node
    ) -> Node:
         
        """Create a matrix multiplication node.

        Parameters
        ----------
        node_A: Node
            The lhs matrix.
        node_B: Node
            The rhs matrix

        Returns
        -------
        result: Node
            The node of the matrix multiplication.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the matrix multiplication result of input values."""
        assert len(input_values) == 2
        return input_values[0] @ input_values[1]
        
        """TODO: your code here"""
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        X = node.inputs[0] 
        Y = node.inputs[1]  

        grad_x = matmul(output_grad, transpose(Y, -2, -1))

        if (len(X.attrs.get("shape", [])) > 2) and (len(Y.attrs.get("shape", [])) == 2):
            batched_grad_y = matmul(transpose(X, -2, -1), output_grad)
            grad_y = sum_op(batched_grad_y, dim=0, keepdim=False)
        else:
            grad_y = matmul(transpose(X, -2, -1), output_grad)

        return [grad_x, grad_y]

#     def gradient(self, node: Node, output_grad: Node) -> List[Node]:
#         X = node.inputs[0]
#         Y = node.inputs[1]
        
#         grad_x = matmul(output_grad, transpose(Y, 0, 1))
#         grad_y = matmul(transpose(X, 0, 1), output_grad)
        
#         return [grad_x, grad_y]
        
#         """Given gradient of matmul node, return partial adjoint to each input."""
#         """TODO: your code here"""


class SoftmaxOp(Op):
    """Softmax operation on input node."""

    def __call__(self, node_A: Node, dim: int = -1) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim},
            name=f"Softmax({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return softmax of input along specified dimension."""
        assert len(input_values) == 1
        x = input_values[0]
        
        dimension = node.attrs["dim"]
        
        return torch.softmax(x, dim=dimension)
        """TODO: your code here"""

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        softmax_out = node
        
        dim = node.attrs['dim']
        sg = mul(softmax_out, output_grad)
        sum_sg = sum_op(sg, dim, keepdim=True)
        diff = sub(output_grad, sum_sg)
        
        grad_input = mul(softmax_out, diff)
        
        return [grad_input]
        
        
        
        """Given gradient of softmax node, return partial adjoint to input."""
        """TODO: your code here"""


class LayerNormOp(Op):
    """Layer normalization operation."""

    def __call__(self, node_A: Node, normalized_shape: List[int], eps: float = 1e-5) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"normalized_shape": normalized_shape, "eps": eps},
            name=f"LayerNorm({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return layer normalized input."""
        assert len(input_values) == 1
        s = input_values[0]
        normalized_shape = node.attrs['normalized_shape']
        epsilon = node.attrs["eps"]
        dims = tuple(range(-len(normalized_shape), 0))
        mu = s.mean(dim=dims, keepdim=True)
        diff = s - mu
        var = (diff * diff).mean(dim=dims, keepdim=True)
        std = torch.sqrt(var + epsilon)
        y = diff / std
        return y
        
        """TODO: your code here"""

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """
        Given gradient of the LayerNorm node wrt its output, return partial 
        adjoint (gradient) wrt the input x.
        """
        s = node.inputs[0]
        normalized_shape = node.attrs["normalized_shape"]
        epsilon = node.attrs["eps"]
        dims = tuple(range(-len(normalized_shape), 0))
        mu = mean(s, dims, keepdim=True)
        diff = sub(s, mu)
        diff_squared = mul(diff, diff)
        var = mean(diff_squared, dims, keepdim=True)
        var_eps = add_by_const(var, epsilon)
        std = sqrt(var_eps)
        y = div(diff, std)
        mean_output_grad = mean(output_grad, dims, keepdim=True)
        output_grad_times_y = mul(output_grad, y)
        mean_output_grad_times_y = mean(output_grad_times_y, dims, keepdim=True)
        numerator = sub(sub(output_grad, mean_output_grad), mul(y, mean_output_grad_times_y))
        grad_x = div(numerator, std)
        return [grad_x]

class ReLUOp(Op):
    """ReLU activation function."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"ReLU({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return ReLU of input."""
        assert len(input_values) == 1
        return torch.relu(input_values[0])


    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of ReLU node, return partial adjoint to input."""
        x = node.inputs[0]
        zero = Node(inputs=[],op=scalar, attrs={"value": 0}, name="0")
        greater_than = greater(x, zero)
        grad = mul(output_grad, greater_than)
        return [grad]


class SqrtOp(Op):
    """Op to compute element-wise square root."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Sqrt({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        x = input_values[0]
        return torch.sqrt(x)
    

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [div(output_grad,mul_by_const(node,2))]


class PowerOp(Op):
    """Op to compute element-wise power."""

    def __call__(self, node_A: Node, exponent: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"exponent": exponent},
            name=f"Power({node_A.name}, {exponent})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        x = input_values[0]
        exp = node.attrs['exponent']
        return torch.pow(x,exp)


    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        x = input_values[0]
        exp = node.attrs['exponent']
        
        x_power = power(x, exp -1)
        
        derivative = mul_by_cons(x_power, exp)
        
        grad = mul(output_grad, derivative)
        return [grad]

class MeanOp(Op):
    """Op to compute mean along specified dimensions.
    
    Note: This is a reference implementation for MeanOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Mean({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        x = input_values[0]
        dim = node.attrs['dim']
        keepdim = node.attrs['keepdim']
        return x.mean(dim=dim, keepdim=keepdim)
    
        """TODO: your code here"""

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        dims = node.attrs['dim']
        x = node.inputs[0]
        input_shape = x.attrs.get("shape",None)
        if input_shape is None:
            raise ValueError(f"Input shape not set for node {x}. Make sure to set the shape (e.g., via Evaluator.run).")
        N = 1
        for i in dims:
            if i < 0:
                i = len(input_shape) + i
            N *= input_shape[i]
        grad_b = broadcast(output_grad, input_shape, input_shape)
        
        grad = mul_by_const(grad_b, 1.0/N)
        return [grad]
            
        
        """TODO: your code here"""


placeholder = PlaceholderOp()
scalar =  NoOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
softmax = SoftmaxOp()
layernorm = LayerNormOp()
relu = ReLUOp()
transpose = TransposeOp()
mean = MeanOp()
sum_op = SumOp()
sqrt = SqrtOp()
power = PowerOp()
greater = GreaterThanOp()
expand_as = ExpandAsOp()
expand_as_3d = ExpandAsOp3d()
log = LogOp()
sub = SubOp()
broadcast = BroadcastOp()

def topological_sort(nodes):
    visited = set()
    ordered = []

    def dfs(n):
        if n in visited:
            return
        visited.add(n)
        for child in n.inputs:
            dfs(child)
        ordered.append(n)
    for n in nodes:
        dfs(n)
    return ordered
    """Helper function to perform topological sort on nodes.
    
    Parameters
    ----------
    nodes : List[Node] or Node
        Node(s) to sort
        
    Returns
    -------
    List[Node]
        Nodes in topological order
    """
    """TODO: your code here"""

class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        """Constructor, which takes the list of nodes to evaluate in the computational graph.

        Parameters
        ----------
        eval_nodes: List[Node]
            The list of nodes whose values are to be computed.
        """
        self.eval_nodes = eval_nodes
        
    def run(self, input_values: Dict[Node, torch.Tensor]) -> List[torch.Tensor]:     
        order = topological_sort(self.eval_nodes)
        computed = {}
        for n in order:
            if n in input_values:
                computed[n] = input_values[n]
                #print(input_values[n].shape)
                n.attrs["shape"] = input_values[n].shape
            else:
                input_vals = []
                for i_node in n.inputs:
                    val = computed[i_node]
                    input_vals.append(val)
                computed[n] = n.op.compute(n, input_vals)
                #print(computed[n].shape)
                n.attrs["shape"] = computed[n].shape
        return [computed[n] for n in self.eval_nodes]
            
        """Computes values of nodes in `eval_nodes` field with
        the computational graph input values given by the `input_values` dict.

        Parameters
        ----------
        input_values: Dict[Node, torch.Tensor]
            The dictionary providing the values for input nodes of the
            computational graph.
            Throw ValueError when the value of any needed input node is
            not given in the dictionary.

        Returns
        -------
        eval_values: List[torch.Tensor]
            The list of values for nodes in `eval_nodes` field.
        """
        """TODO: your code here"""


def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    topological = topological_sort([output_node])
    grads = {}
    grads[output_node] = ones_like(output_node)
    
    for node in reversed(topological):
        if node not in grads:
            continue
        grad = grads[node]
        if isinstance(node.op, PlaceholderOp):
            continue
        input_grad = node.op.gradient(node, grad)
        for i, i_node in enumerate(node.inputs):
            if i_node in grads:
                grads[i_node] = add(grads[i_node], input_grad[i])
            else:
                grads[i_node] = input_grad[i]
    output = []
    for n in nodes:
        if n in grads:
            output.append(grads[n])
        else:
            output.append(zeros_like(n))
    return output


    

