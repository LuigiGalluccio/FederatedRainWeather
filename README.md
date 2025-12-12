# Federated Learning for Distributed Weather Forecasting: A Practical Approach on Real Multidimensional Georeferenced Data

In this work, we explore a federated learning framework designed for collaborative weather forecasting using distribuited meteorological data. By combining edge computing, privacy-preserving learning and advanced deep learning architectures such as Transformers and Crossformers, we demonstrate improved prediction performance while maintaining data confidentiality and reducing communication overhead.

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
cd FederatedWeatherCC
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
## Dataset layout
Inside the directory:
```bash
cd data/processed_data/
```
the dataset includes **seven features**:
* ```datetime```
* ```station_id```
* ```temperature```
* ```humidity```
* ```wind_speed```
* ```wind_sin```
* ```wind_cos```

```wind_sin``` and ```wind_cos``` represent a sinusoidal encoding of wind direction
## Training Example

Models can be trained in two different modes:
* **standard (non-federated) training**
* **federated learning**

---

#### Standard training

To run a standard training (non-federated) training session:

1. Navigate to the directory of the model you want to train:

   * `src/model/transformer/` for the Transformer model
   * `src/model/crossformer/` for the Crossformer model

2. Launch the training script:

```bash
cd src/model/transformer  # or src/model/crossformer
python train.py
```

---

#### With Federated Learning

To run the project in a federated learning setting:

1. Open **multiple terminal windows** - one for the server, and one for each client, with a minimum of 4 clients.

2. In every terminal, navigate to the directory of the model you want to use:

   * `src/model/transformer/`
   * `src/model/crossformer/`

3. In the **first terminal**, start the federated learning **server**:

```bash
cd src/model/transformer  # or src/model/crossformer
python server.py
```

4. In the **clients terminals**, launch a**client** for each terminal:

```bash
cd src/model/transformer  # or src/model/crossformer
python client.py <place_id> <server_address> # python client.py <id> localhost:8081
```
