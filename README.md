
# Conservative to Primitive Variable Recovery in GRMHD Simulations

This project focuses on machine learning for recovering primitive variables from conserved quantities in GRMHD simulations. The model used in this project is a feedforward neural network (FNN), optimized using various hyperparameters. The project demonstrates how to preprocess data, train a model, and make predictions.

## Project Structure

```
conservative-to-primitive/
├── Balsara1_shocktube/              # Root directory for simulation data
│   ├── Balsara1_shocktube_xdir_WZ/  # Folder containing the simulation output
│   │   ├── output-0000/
│   │   │   ├── conservatives/       # Folder containing conserved variables
│   │   │   │   ├── db-*.tsv        # Conserved variable files
│   │   │   │   ├── dens-*.tsv
│   │   │   │   ├── mom-*.tsv
│   │   │   │   └── tau-*.tsv
│   │   │   ├── primitives/         # Folder containing primitive variables
│   │   │   │   ├── rho-*.tsv       # Primitive variable files
│   │   │   │   ├── vel-*.tsv
│   │   │   │   ├── eps-*.tsv
│   │   │   │   ├── press-*.tsv
│   │   │   │   └── bvec-*.tsv
├── data_loader.py                  # Loads and preprocesses simulation data
├── fnn.py                          # Contains the FNN model architecture
├── hyperparam_optimization.py      # Hyperparameter optimization using Optuna
├── main.py                         # Main script to run the model training and evaluation
├── trainer.py                      # Defines the ModelTrainer class for training and testing
├── cuda_check.py                   # Handles GPU/CPU device selection
├── pyproject.toml                  # Poetry dependencies and project configuration
└── README.md                       # Project documentation
```

## Dependencies

This project uses **Poetry** for dependency management. The required packages are listed in the `pyproject.toml` file. You can install them by running:

```bash
poetry install
```

The project depends on the following key packages:

- `torch` for deep learning models
- `pandas` for data manipulation
- `numpy` for numerical operations
- `scikit-learn` for machine learning utilities (e.g., scaling, splitting data)
- `optuna` for hyperparameter optimization
- `matplotlib`, `seaborn`, and `plotly` for data visualization
- `tensorrt` and `onnx` for deployment optimizations (if needed)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/ParthS33/ml-grmhd-primitive-recovery.git
    cd ml-grmhd-primitive-recovery
    ```

2. Install the dependencies:

    ```bash
    poetry install
    ```

3. Set up your Python environment and dependencies using Poetry.

## Data

The data used in this project is from GRMHD simulations and includes **conserved** and **primitive** variables.

- **Conserved variables** include: `db`, `dens`, `mom`, `tau`
- **Primitive variables** include: `rho`, `vel`, `eps`, `press`, `bvec`

The data is located in the following directories:

- `Balsara1_shocktube/Balsara1_shocktube_xdir_WZ/output-0000/Balsara1_shocktube_xdir/conservatives/` (for conserved variables)
- `Balsara1_shocktube/Balsara1_shocktube_xdir_WZ/output-0000/Balsara1_shocktube_xdir/primitives/` (for primitive variables)

Ensure that the simulation output files are correctly placed in these directories for the data loading and merging process.

## Running the Model

### Basic Model Training

To train the basic FNN model, you can use the `run_basic()` function in the `main.py` script:

```bash
python main.py
```

This will:
- Initialize the FNN model with two hidden layers (600 and 450 units)
- Train the model on the provided data
- Evaluate the model performance

### DAIN Model

To use the DAIN (Dynamic Adaptive Inference Networks) model, call the `run_dain()` function:

```bash
python main.py
```

This will:
- Initialize the DAIN model with 3 blocks
- Train the model and evaluate its performance

### Hyperparameter Optimization

To optimize the model's hyperparameters, the `run_hyperparameter_tuning()` function can be used. This employs Optuna to find the best hyperparameters for the model training.

```bash
python main.py
```

## Hyperparameters

The model's key hyperparameters are:
- `learning_rate` (e.g., 3e-4, 1e-4)
- `batch_size` (e.g., 32, 64)
- `epochs` (e.g., 85)
- `hidden layer units` (for FNN models)

Hyperparameter optimization can be performed by running the `optimize()` method in `hyperparam_optimization.py`.

## Model Evaluation

Once trained, the model can be evaluated using the `test_model()` method in `trainer.py`. The test results include:
- Loss metrics
- Average inference time per sample

## CUDA Support

This project supports CUDA-enabled GPUs for model training and inference. The `cuda_check.py` script ensures that the model is running on the available GPU or CPU.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project leverages Optuna for hyperparameter optimization.
- Special thanks to the developers of PyTorch and other dependencies.
