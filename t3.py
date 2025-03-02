import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28  # MNIST images are 28x28

def transformer(X: ad.Node, nodes: List[ad.Node],
                model_dim: int, seq_length: int, eps, batch_size, num_classes) -> ad.Node:
    # Unpack parameters
    w_q, w_k, w_v, w_o, w_1, w_2, b_1, b_2 = nodes

    # Self-Attention
    Q = ad.matmul(X, w_q)
    K = ad.matmul(X, w_k)
    V = ad.matmul(X, w_v)
    K_t = ad.transpose(K, 1, 2)
    scores = ad.matmul(Q, K_t)
    scale = model_dim ** 0.5
    scaled_scores = ad.div_by_const(scores, scale)
    attn_weights = ad.softmax(scaled_scores, dim=-1)
    context = ad.matmul(attn_weights, V)
    attn_output = ad.matmul(context, w_o)

    # Pooling
    pooled = ad.sum_op(attn_output, dim=1, keepdim=False)
    avg_pooled = ad.div_by_const(pooled, seq_length)

    # Feed-Forward Network
    hidden_linear = ad.matmul(avg_pooled, w_1)
    hidden_linear = ad.add(hidden_linear, b_1)
    hidden = ad.relu(hidden_linear)
    logits_linear = ad.matmul(hidden, w_2)
    logits = ad.add(logits_linear, b_2)
    return logits

def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    probs = ad.softmax(Z, dim=1)
    log_probs = ad.log(probs)
    prod = ad.mul(y_one_hot, log_probs)
    loss_per_example = ad.sum_op(prod, dim=1, keepdim=False)
    total_loss = ad.sum_op(loss_per_example, dim=0, keepdim=False)
    avg_loss = ad.div_by_const(ad.mul_by_const(total_loss, -1), batch_size)
    return avg_loss

def sgd_epoch(f_run_model: Callable,
              X: torch.Tensor,
              y: torch.Tensor,
              model_weights: List[torch.Tensor],
              batch_size: int,
              lr: float) -> Tuple[List[torch.Tensor], float]:
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size
    total_loss = 0.0
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_examples)
        if end_idx - start_idx == 0:
            continue
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        outputs = f_run_model(model_weights, X_batch, y_batch)
        loss_val = outputs[1]
        grads = outputs[2:]
        for j in range(len(model_weights)):
            model_weights[j] = model_weights[j] - lr * grads[j]
        total_loss += loss_val.item() * (end_idx - start_idx)
    average_loss = total_loss / num_examples
    print("Avg_loss:", average_loss)
    return model_weights, average_loss

def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # TODO: Tune your hyperparameters here
    input_dim = 28  # Each row of the MNIST image
    max_len = 28    # Number of rows in the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10
    model_dim = 128
    eps = 1e-5 

    # - Set up the training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # TODO: Define the forward graph.
    X_node = ad.Variable(name="X")
    y_groundtruth = ad.Variable(name="y")
    W_Q = ad.Variable(name="W_Q")
    W_K = ad.Variable(name="W_K")
    W_V = ad.Variable(name="W_V")
    W_O = ad.Variable(name="W_O")
    W_1 = ad.Variable(name="W_1")
    W_2 = ad.Variable(name="W_2")
    b_1 = ad.Variable(name="b_1")
    b_2 = ad.Variable(name="b_2")
    param_nodes = [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]

    y_predict: ad.Node = transformer(
        X_node, param_nodes, model_dim, seq_length, eps, batch_size, num_classes
    )
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)
    
    # --- IMPORTANT: Run a dummy forward pass to set input shapes ---
    # Create dummy model weights using the known shapes.
    dummy_model_weights = [
        torch.zeros(input_dim, model_dim),      # W_Q
        torch.zeros(input_dim, model_dim),      # W_K
        torch.zeros(input_dim, model_dim),      # W_V
        torch.zeros(model_dim, model_dim),      # W_O
        torch.zeros(model_dim, model_dim),      # W_1
        torch.zeros(model_dim, num_classes),    # W_2
        torch.zeros(model_dim),                 # b_1
        torch.zeros(num_classes),               # b_2
    ]
    dummy_feed_dict = {
        X_node: torch.zeros(1, seq_length, input_dim),
        y_groundtruth: torch.zeros(1, num_classes),
        W_Q: dummy_model_weights[0],
        W_K: dummy_model_weights[1],
        W_V: dummy_model_weights[2],
        W_O: dummy_model_weights[3],
        W_1: dummy_model_weights[4],
        W_2: dummy_model_weights[5],
        b_1: dummy_model_weights[6],
        b_2: dummy_model_weights[7],
    }
    temp_eval = ad.Evaluator([loss])
    _ = temp_eval.run(dummy_feed_dict)
    # -----------------------------------------------------------------

    # TODO: Construct the backward graph.
    grads: List[ad.Node] = ad.gradients(loss, param_nodes)
    
    # TODO: Create the evaluator.
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    # Initialize model weights.
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

    def f_run_model(model_weights, X_batch, y_batch):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        result = evaluator.run(
            input_values={
                X_node: X_batch,
                y_groundtruth: y_batch,
                W_Q: model_weights[0],
                W_K: model_weights[1],
                W_V: model_weights[2],
                W_O: model_weights[3],
                W_1: model_weights[4],
                W_2: model_weights[5],
                b_1: model_weights[6],
                b_2: model_weights[7],
            }
        )
        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size
        all_logits = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_examples)
            if end_idx - start_idx == 0:
                continue
            X_batch = X_val[start_idx:end_idx, :seq_length]
            feed_dict = {
                X_node: X_batch,
                W_Q: model_weights[0],
                W_K: model_weights[1],
                W_V: model_weights[2],
                W_O: model_weights[3],
                W_1: model_weights[4],
                W_2: model_weights[5],
                b_1: model_weights[6],
                b_2: model_weights[7],
            }
            logits = test_evaluator.run(feed_dict)[0]
            all_logits.append(logits)
        concatenated_logits = np.concatenate([log.detach().numpy() for log in all_logits], axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Convert datasets to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    # Prepare initial model weights as torch tensors.
    model_weights: List[torch.Tensor] = [
        torch.tensor(W_Q_val, dtype=torch.float32),
        torch.tensor(W_K_val, dtype=torch.float32),
        torch.tensor(W_V_val, dtype=torch.float32),
        torch.tensor(W_O_val, dtype=torch.float32),
        torch.tensor(W_1_val, dtype=torch.float32),
        torch.tensor(W_2_val, dtype=torch.float32),
        torch.tensor(b_1_val, dtype=torch.float32),
        torch.tensor(b_2_val, dtype=torch.float32)
    ]

    # Train the model.
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(f_run_model, X_train, y_train, model_weights, batch_size, lr)
        predict_label = f_eval_model(X_test, model_weights)
        print("test")
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label == y_test_tensor.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test_tensor.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")