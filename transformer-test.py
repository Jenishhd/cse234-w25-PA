import math
from typing import Callable, Tuple, List

import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import torch
from torchvision import datasets, transforms

import auto_diff as ad

max_len = 28  # For MNIST, each image is 28x28

##############################################################################
# Transformer Forward Graph
##############################################################################
def transformer(
    X: ad.Node,
    nodes: List[ad.Node],
    model_dim: int,
    seq_length: int,
    eps,
    batch_size,
    num_classes: int,
) -> ad.Node:
    """
    Constructs the computational graph for a single transformer layer with sequence classification.

    The weight nodes (in order) are assumed to be:
      0: W_Q, shape (input_dim, model_dim)
      1: W_K, shape (input_dim, model_dim)
      2: W_V, shape (input_dim, model_dim)
      3: W_O, shape (model_dim, model_dim)
      4: W_1, shape (model_dim, model_dim)
      5: W_2, shape (model_dim, num_classes)
      6: b_1, shape (model_dim,)    (for feed–forward first layer)
      7: b_2, shape (num_classes,)   (for feed–forward second layer)
    """
    W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = nodes

    # ---- Single-Head Self-Attention ----
    # Compute query, key, and value projections.
    # (Assume X has shape (batch, seq_length, input_dim) and input_dim == 28.)
    Q = ad.matmul(X, W_Q)   # shape: (batch, seq_length, model_dim)
    K = ad.matmul(X, W_K)   # shape: (batch, seq_length, model_dim)
    V = ad.matmul(X, W_V)   # shape: (batch, seq_length, model_dim)

    # To compute attention, transpose K along the last two dims:
    # (batch, seq_length, model_dim) -> (batch, model_dim, seq_length)
    K_T = ad.transpose(K, 1, 2)
    # Compute dot-product attention logits: (batch, seq_length, seq_length)
    attn_logits = ad.matmul(Q, K_T)
    # Scale by sqrt(model_dim) to avoid overly large values.
    scale = math.sqrt(model_dim)
    attn_logits_scaled = ad.div_by_const(attn_logits, scale)
    # Softmax along the last dimension (the key dimension)
    attn_weights = ad.softmax(attn_logits_scaled, dim=2)
    # Compute the weighted sum over the values: (batch, seq_length, model_dim)
    attn_output = ad.matmul(attn_weights, V)

    # ---- Post-Attention Projection ----
    # Apply a learned linear transformation.
    attn_proj = ad.matmul(attn_output, W_O)

    # ---- Feed-Forward Network ----
    # First linear layer (with bias) followed by ReLU.
    ffn_hidden = ad.add(ad.matmul(attn_proj, W_1), b_1)
    ffn_hidden_relu = ad.relu(ffn_hidden)
    # Second linear layer (with bias) that outputs logits per token.
    logits = ad.add(ad.matmul(ffn_hidden_relu, W_2), b_2)

    # For sequence classification, average over the sequence dimension.
    output = ad.mean(logits, dim=(1,), keepdim=False)  # shape: (batch, num_classes)
    return output

##############################################################################
# Softmax Loss Graph
##############################################################################
def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """
    Constructs the computational graph of average softmax loss.

    The loss is defined as:
      loss = - (1 / batch_size) * sum_i sum_j y[i,j] * log( softmax(Z)[i,j] )
    """
    # Compute the softmax probabilities.
    probs = ad.softmax(Z, dim=1)
    # Take the logarithm.
    log_probs = ad.log(probs)
    # Only the true class (where y_one_hot == 1) contributes.
    loss_elements = ad.mul(y_one_hot, log_probs)
    # Sum over the class dimension for each example.
    loss_per_example = ad.sum_op(loss_elements, dim=(1,), keepdim=False)
    # Negative log-likelihood.
    nll = ad.mul_by_const(loss_per_example, -1)
    # Total loss is the sum over the batch, then average.
    total_loss = ad.sum_op(nll, dim=(0,), keepdim=False)
    loss = ad.div_by_const(total_loss, batch_size)
    return loss

##############################################################################
# SGD Epoch Function
##############################################################################
def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> Tuple[List[torch.Tensor], float]:
    """
    Runs one epoch of SGD.

    f_run_model should return a list:
      [y_predict, loss, grad_W_Q, grad_W_K, grad_W_V, grad_W_O, grad_W_1, grad_W_2, grad_b_1, grad_b_2]
    """
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size
    total_loss = 0.0

    for i in range(num_batches):
        start_idx = i * batch_size
        if start_idx + batch_size > num_examples:
            continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        # Run forward and backward passes.
        outputs = f_run_model(model_weights, X_batch, y_batch)
        # Unpack outputs: first is prediction, then loss, then gradients.
        y_predict, loss_val, *grads = outputs

        total_loss += loss_val.item() * (end_idx - start_idx)

        # Update each model parameter with an SGD step.
        for j in range(len(model_weights)):
            # Here we assume grads[j] is already the gradient tensor for the j-th parameter.
            model_weights[j] = model_weights[j] - lr * grads[j]

    average_loss = total_loss / num_examples
    print("Avg_loss:", average_loss)
    return model_weights, average_loss

##############################################################################
# Training Routine
##############################################################################
def train_model():
    """
    Train a transformer model on the MNIST dataset using our auto-diff framework.
    """
    # --- Hyperparameters ---
    input_dim = 28         # Each row of MNIST image
    seq_length = max_len   # Number of rows (i.e. sequence length)
    num_classes = 10
    model_dim = 128
    eps = 1e-5
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # --- Define the Computational Graph ---
    # Input and ground-truth variable nodes.
    x = ad.Variable("x")              # Expected shape: (batch, seq_length, input_dim)
    y_groundtruth = ad.Variable("y")  # Expected shape: (batch, num_classes)

    # Create weight nodes as placeholders.
    W_Q = ad.Variable("W_Q")
    W_K = ad.Variable("W_K")
    W_V = ad.Variable("W_V")
    W_O = ad.Variable("W_O")
    W_1 = ad.Variable("W_1")
    W_2 = ad.Variable("W_2")
    b_1 = ad.Variable("b_1")
    b_2 = ad.Variable("b_2")
    weight_nodes = [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]

    # Build the forward graph.
    y_predict = transformer(x, weight_nodes, model_dim, seq_length, eps, batch_size, num_classes)
    loss = softmax_loss(y_predict, y_groundtruth, batch_size)
    # --- Dummy forward pass to set shapes ---
    dummy_input = torch.zeros((batch_size, seq_length, input_dim), dtype=torch.float32)
    dummy_y = torch.zeros((batch_size, num_classes), dtype=torch.float32)
    dummy_mapping = {
        x: dummy_input,
        y_groundtruth: dummy_y,
        W_Q: torch.zeros((input_dim, model_dim), dtype=torch.float32),
        W_K: torch.zeros((input_dim, model_dim), dtype=torch.float32),
        W_V: torch.zeros((input_dim, model_dim), dtype=torch.float32),
        W_O: torch.zeros((model_dim, model_dim), dtype=torch.float32),
        W_1: torch.zeros((model_dim, model_dim), dtype=torch.float32),
        W_2: torch.zeros((model_dim, num_classes), dtype=torch.float32),
        b_1: torch.zeros((model_dim,), dtype=torch.float32),
        b_2: torch.zeros((num_classes,), dtype=torch.float32),
    }
    temp_eval = ad.Evaluator([loss])
    _ = temp_eval.run(dummy_mapping)

    # Construct the backward graph by computing gradients with respect to weight nodes.
    grads = ad.gradients(loss, weight_nodes)

    # Create evaluators.
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # --- Data Loading and Preprocessing ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset  = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Reshape images: (num_examples, 28, 28) and normalize.
    X_train = train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_test = test_dataset.targets.numpy()

    # One-hot encode labels.
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot  = encoder.transform(y_test.reshape(-1, 1))

    # --- Initialize Model Weights ---
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    model_weights = [
        torch.tensor(W_Q_val, dtype=torch.float32),
        torch.tensor(W_K_val, dtype=torch.float32),
        torch.tensor(W_V_val, dtype=torch.float32),
        torch.tensor(W_O_val, dtype=torch.float32),
        torch.tensor(W_1_val, dtype=torch.float32),
        torch.tensor(W_2_val, dtype=torch.float32),
        torch.tensor(b_1_val, dtype=torch.float32),
        torch.tensor(b_2_val, dtype=torch.float32),
    ]

    # --- Define f_run_model ---
    def f_run_model(model_weights: List[torch.Tensor], X_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        Runs both forward and backward passes.
        Maps each auto-diff variable to its current tensor value.
        """
        input_mapping = {
            x: X_batch,
            y_groundtruth: torch.tensor(y_batch, dtype=torch.float32),
            W_Q: model_weights[0],
            W_K: model_weights[1],
            W_V: model_weights[2],
            W_O: model_weights[3],
            W_1: model_weights[4],
            W_2: model_weights[5],
            b_1: model_weights[6],
            b_2: model_weights[7],
        }
        return evaluator.run(input_mapping)

    # --- Define f_eval_model ---
    def f_eval_model(X_val: torch.Tensor, model_weights: List[torch.Tensor]):
        """
        Runs only the forward pass and returns predictions.
        """
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size
        all_logits = []
        for i in range(num_batches):
            start_idx = i * batch_size
            if start_idx + batch_size > num_examples:
                continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx]
            input_mapping = {
                x: X_batch,
                W_Q: model_weights[0],
                W_K: model_weights[1],
                W_V: model_weights[2],
                W_O: model_weights[3],
                W_1: model_weights[4],
                W_2: model_weights[5],
                b_1: model_weights[6],
                b_2: model_weights[7],
            }
            logits = test_evaluator.run(input_mapping)
            all_logits.append(logits[0])
        # Concatenate logits from all batches.
        concatenated_logits = np.concatenate(
            [t.detach().numpy() for t in all_logits], axis=0
        )
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # --- Convert data to torch.Tensors ---
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_onehot, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_onehot, dtype=torch.float32)

    # --- Training Loop ---
    for epoch in range(num_epochs):
        X_train_tensor, y_train_tensor = shuffle(X_train_tensor, y_train_tensor)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train_tensor, y_train_tensor, model_weights, batch_size, lr
        )
        pred_labels = f_eval_model(X_test_tensor, model_weights)
        accuracy = np.mean(pred_labels == y_test)
        print(f"Epoch {epoch}: test accuracy = {accuracy:.4f}, loss = {loss_val}")

    final_pred = f_eval_model(X_test_tensor, model_weights)
    final_accuracy = np.mean(final_pred == y_test)
    return final_accuracy

##############################################################################
# Main
##############################################################################
if __name__ == "__main__":
    acc = train_model()
    print(f"Final test accuracy: {acc}")

