# Graph Neural Network for Node Classification in a Citation Network

This repository contains the source code for a Graph Convolutional Network (GCN) designed for node classification. The project implements a GCN model in **PyTorch Geometric** to predict the academic subfield of research papers in the **Cora citation network**.



---

## Abstract
This project demonstrates the application of Graph Neural Networks (GNNs) to a common problem in network science: semi-supervised node classification. By leveraging the citation links (graph structure) between academic papers, a GCN model is trained to infer the topic of a paper more effectively than traditional methods that only consider the paper's content. This work showcases skills in graph-based machine learning, a cutting-edge area of AI, and implements a standard research baseline on the Cora dataset.

---

## 1. Model Architecture: Graph Convolutional Network (GCN)

The model is a **Graph Convolutional Network (GCN)**, a powerful type of GNN designed for tasks on graph-structured data.

Unlike traditional neural networks that assume independent data points, a GCN leverages the connections between nodes. It works through a process called **message passing**, where each node in the graph iteratively updates its feature vector by aggregating information from its immediate neighbors. After a few layers, each node's representation is enriched with information about its local neighborhood, allowing the model to make highly context-aware predictions.

Our implementation consists of two `GCNConv` layers with a `ReLU` activation and `Dropout` for regularization.

---

## 2. Dataset: Cora

The model is trained on the **Cora dataset**, a standard benchmark for GNNs.
* **Nodes:** 2,708 scientific publications.
* **Edges:** 5,429 citation links between them.
* **Node Features:** A 1,433-dimensional binary vector for each paper, indicating the presence or absence of words from a dictionary.
* **Labels:** Each paper is classified into one of seven research subfields (e.g., "Neural Networks", "Reinforcement Learning").

The task is semi-supervised, meaning the model is trained on a small fraction of labeled nodes and must predict the labels for the rest.

---

## 3. Results & Analysis

**[RESULTS PENDING]**
The model will be trained for 200 epochs. The primary evaluation metric is classification accuracy on the held-out test set of nodes.


### ### Qualitative Analysis
* **Why GNNs Excel:** The analysis will compare the GCN's performance to a simple Multi-Layer Perceptron (MLP) that ignores the graph structure, demonstrating the value of leveraging relational information.
* **Error Analysis:** The model's misclassifications will be examined. It is expected that papers at the intersection of multiple fields (e.g., a paper on "Genetic Algorithms" which is cited by both "Neural Networks" and "Theory" papers) will be the most challenging to classify correctly.
    ---

## 4. How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AliJalilpourMoones/gnn-citation-network-portfolio.git](https://github.com/AliJalilpourMoones/gnn-citation-network-portfolio.git)
    cd gnn-citation-network-portfolio
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    **Important:** PyTorch Geometric often requires a specific installation command based on your system's PyTorch and CUDA versions.
    * First, install PyTorch by following the official instructions: [https://pytorch.org/](https://pytorch.org/)
    * Then, find the correct installation command for `torch_geometric` here: [https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
    * Finally, install any remaining packages: `pip install tqdm`

4.  **Run the training script:**
    ```bash
    python src/main.py
    ```