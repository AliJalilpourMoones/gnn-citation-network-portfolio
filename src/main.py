# src/main.py

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from tqdm import tqdm

# Import from our other project files
from config import Config
from model import GCN

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    # Perform a single forward pass
    out = model(data.x, data.edge_index)
    # Compute the loss using the training nodes
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    
    # Check accuracy on train, val, and test masks
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = (pred[mask] == data.y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs

def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load the Cora dataset ---
    # PyTorch Geometric will automatically download it for you!
    print("Loading Cora dataset...")
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0].to(device)

    # --- 2. Initialize the Model ---
    print("Initializing GCN model...")
    model = GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=config.HIDDEN_CHANNELS,
        out_channels=dataset.num_classes,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # --- 3. Training Loop ---
    print("Starting training...")
    for epoch in range(1, config.EPOCHS + 1):
        loss = train(model, data, optimizer)
        if epoch % 10 == 0:
            train_acc, val_acc, test_acc = test(model, data)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    # --- 4. Final Evaluation ---
    train_acc, val_acc, test_acc = test(model, data)
    print("\nTraining complete.")
    print(f"Final Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()