import matplotlib.pyplot as plt
import numpy as np
import psutil
import time
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from tensorflow.keras.optimizers import RMSprop
from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_siamese_network
from tensorflow.keras import backend as K
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ✅ Enable GPU & Prevent Out-of-Memory (OOM) Errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Prevents full allocation
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)

# ✅ Define Contrastive Loss (Baseline)
def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = K.cast(y_true, y_pred.dtype)
    positive_loss = y_true * K.square(y_pred)  # Genuine pairs
    negative_loss = (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))  # Forged pairs
    return K.mean(positive_loss + negative_loss + K.epsilon())  # Prevent zero errors

# ✅ Dataset Configuration
datasets = {
    "BHSig260_Hindi": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Hindi",
        "train_writers": list(range(101, 169)),
        "test_writers": list(range(170, 260))
    }
}

# ✅ Function to Load Data
def load_data(dataset_name, dataset_config):
    generator = SignatureDataGenerator(
        dataset={
            dataset_name: {
                "path": dataset_config["path"],
                "train_writers": dataset_config["train_writers"],
                "test_writers": dataset_config["test_writers"]
            }
        },
        img_height=155,
        img_width=220,
        apply_esrgan=False  # No ESRGAN in main.py
    )
    
    train_data, train_labels = generator.get_train_data()
    test_data, test_labels = generator.get_test_data()

    return train_data, train_labels, test_data, test_labels

# ✅ Compute Class Imbalance
def compute_class_distribution(train_labels, test_labels, dataset_name):
    train_counts = np.bincount(train_labels)
    test_counts = np.bincount(test_labels)

    print(f"\n--- Class Imbalance in {dataset_name} ---")
    print(f"Training Data - Genuine: {train_counts[1] if len(train_counts) > 1 else 0}, Forged: {train_counts[0]}")
    print(f"Test Data - Genuine: {test_counts[1] if len(test_counts) > 1 else 0}, Forged: {test_counts[0]}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Genuine", "Forged"], train_counts, color=["blue", "red"])
    axes[0].set_title(f"Train Data - {dataset_name}")
    axes[0].set_xlabel("Signature Type")
    axes[0].set_ylabel("Count")

    axes[1].bar(["Genuine", "Forged"], test_counts, color=["blue", "red"])
    axes[1].set_title(f"Test Data - {dataset_name}")
    axes[1].set_xlabel("Signature Type")

    plt.suptitle(f"Class Distribution for {dataset_name}")
    plt.tight_layout()
    plt.show()

# ✅ Compute Scalability Metrics
def compute_scalability_metrics(start_time, end_time, dataset_name):
    execution_time = end_time - start_time
    memory_usage = psutil.virtual_memory().percent  # Get memory usage
    print(f"\n--- Scalability Metrics for {dataset_name} ---")
    print(f"Training & Evaluation Time: {execution_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f}%")

# ✅ Compute Noise Sensitivity
def compute_noise_sensitivity(y_true, y_pred, dataset_name):
    y_pred_labels = (y_pred > 0.5).astype(int)  # Convert probabilities to binary
    accuracy = accuracy_score(y_true, y_pred_labels)
    precision = precision_score(y_true, y_pred_labels, zero_division=1)
    recall = recall_score(y_true, y_pred_labels, zero_division=1)
    f1 = f1_score(y_true, y_pred_labels)

    auc = roc_auc_score(y_true, y_pred.astype(float))  # Ensure float for ROC AUC

    gar = recall  # Genuine Acceptance Rate
    frr = 1 - gar  # False Rejection Rate
    far = 1 - precision  # False Acceptance Rate

    print("\n--- Noise Sensitivity Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (GAR): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Genuine Acceptance Rate (GAR): {gar:.4f}")
    print(f"False Rejection Rate (FRR): {frr:.4f}")
    print(f"False Acceptance Rate (FAR): {far:.4f}")

# ✅ Evaluate Model on GPU
def evaluate_model_on_gpu(model, test_data):
    with tf.device('/GPU:0'):  # Explicitly set to GPU
        y_pred = model.predict(test_data, batch_size=16)
    return y_pred

# ✅ Main Script
for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Processing Dataset: {dataset_name} ---")

    # Load dataset
    train_data, train_labels, test_data, test_labels = load_data(dataset_name, dataset_config)

    # ✅ Compute Class Distribution
    compute_class_distribution(train_labels, test_labels, dataset_name)

    # Create and compile model
    model = create_siamese_network(input_shape=(155, 220, 1))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss=contrastive_loss)

    # Train model on GPU
    print(f"Training on {dataset_name}...")
    start_time = time.time()
    with tf.device('/GPU:0'):  # Run training on GPU
        model.fit(train_data, train_labels, epochs=5, batch_size=8, validation_split=0.2, verbose=1)
    end_time = time.time()

    # Evaluate model
    print(f"Evaluating on {dataset_name}...")
    y_pred = evaluate_model_on_gpu(model, test_data)  # ✅ Move inference to GPU

    # ✅ Compute Scalability & Noise Sensitivity Metrics
    compute_scalability_metrics(start_time, end_time, dataset_name)
    compute_noise_sensitivity(test_labels, y_pred, dataset_name)
