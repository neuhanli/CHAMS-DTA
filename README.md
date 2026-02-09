This project implements a deep learning model for Drug-Target Interaction (DTI) prediction, combining cross-modal attention and multi-stage sampling mechanisms.

Key Features
Dual-Modal Processing: Separate processing of drug SMILES sequences and protein amino acid sequences.

Hierarchical Attention Mechanism: Combines cross-modal attention with local attention for feature selection.

Multi-Stage Sampling: Progressive feature filtering through multi-stage attention mechanisms.

Convolutional Attention Blocks: Uses CABlock for feature enhancement.

Project Structure
text
├── create_data.py      # Data preprocessing and dataset creation
├── model.py           # Main model definition
├── training.py        # Training and evaluation scripts
├── utils.py           # Utility functions and dataset classes
├── requirements.txt   # Required packages
├── data/              # Data directory
│   ├── davis/         # DAVIS dataset
│   └── ...
└── README.md          # Project documentation
Installation
Ensure you have Python installed.

Install the required packages:

bash
pip install -r requirements.txt
Usage
1. Data Preparation
Ensure your data files are organized in the following structure under the project root:

text
data/
└── davis/
    ├── ligands_can.txt
    ├── proteins.txt
    ├── Y
    └── folds/
        ├── train_fold_setting1.txt
        └── test_fold_setting1.txt
2. Training
Execute the main training script:

bash
python training.py
Note: Running training.py will first check and preprocess the data if it hasn't been done, and then proceed to train the model.

3. Configuration
Key parameters can be adjusted within the training.py script:

TRAIN_BATCH_SIZE: Training batch size (default: 256)

TEST_BATCH_SIZE: Testing batch size (default: 256)

LR: Learning rate (default: 0.0001)

NUM_EPOCHS: Number of training epochs (default: 1000)


Evaluation Metrics
The model is evaluated using the following metrics:

MSE (Mean Squared Error)

CI (Concordance Index)

RM2 (Adjusted R-squared)
