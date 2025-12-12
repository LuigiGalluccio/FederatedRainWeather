# Federated Learning for Distributed Weather Forecasting: A Practical Approach on Real Multidimensional Georeferenced Data

FederatedWeather is a machine learning project focused on weather forecasting using federated learning techniques and transformer-based models. This project aims to preprocess distributed weather datasets, train models in a decentralized setting, and evaluate their performance.

---

## Project Structure

```
FederatedWeather/
├── data/
│   ├── processed_data/        # Cleaned and formatted datasets ready for training
│   └── trained_model/         # Saved models and checkpoints after training
│
├── src/
│   ├── model/                 # Core machine learning model code
│   │   ├── crossformer/       # Crossformer-based model architectures and layers
│   │   └── transformer/       # Transformer-based model architectures and layers
│   └── utils/                 # Utility scripts and helper functions (e.g., logging, metrics)
│
├── .gitignore                 # Git ignore file
├── LICENSE                    # Project license
├── README.md                  # Project overview and documentation
└── requirements.txt           # Project dependencies
```

---
## Getting Started

To set up and run the project locally, follow these steps:

### 1. Enter the repository

```bash
cd FederatedWeather
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the Dependencies

```bash
pip install -r requirements.txt
```

### **Run the Project**

You can run the project in two modes: **standard training** (non-federated) and **federated learning**.

---

#### Without Federated Learning

To run a standard training session (without federated learning):

1. Navigate to the desired model directory:

   * `src/model/transformer/` for the Transformer-based model
   * `src/model/crossformer/` for the Crossformer-based model

2. Run `train.py`:

```bash
cd src/model/transformer  # or src/model/crossformer
python train.py
```

---

#### With Federated Learning

To run the project using Federated Learning:

1. Open **multiple terminal windows** (one for the server, and one for each client).

2. In each terminal, navigate to either:

   * `src/model/transformer/`
   * or `src/model/crossformer/`

3. In the **first terminal**, start the federated learning **server**:

```bash
cd src/model/transformer  # or src/model/crossformer
python server.py
```

4. In the **other terminals**, start one or more **clients**:

```bash
cd src/model/transformer  # or src/model/crossformer
python client.py
```
