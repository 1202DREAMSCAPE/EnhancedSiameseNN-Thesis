import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import faiss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_siamese_network
from tensorflow.keras.models import load_model

# Ensure reproducibility
import random
np.random.seed(1337)
random.seed(1337)

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = K.sum(K.square(anchor - positive), axis=-1)
    neg_dist = K.sum(K.square(anchor - negative), axis=-1)
    loss = K.maximum(pos_dist - neg_dist + alpha, 0.0)
    return K.mean(loss)

def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def search_faiss(index, query_vectors, k=5):
    query_vectors = np.array(query_vectors, dtype=np.float32)
    distances, indices = index.search(query_vectors, k)
    return distances, indices

def log_metrics(start_time, description=""):
    end_time = time.perf_counter()
    memory_usage = (torch.cuda.memory_allocated() / (1024 * 1024)
                    if torch.cuda.is_available() else 0)  # GPU memory
    print(f"🔹 {description} - Time: {end_time - start_time:.4f}s | GPU Memory: {memory_usage:.2f} MB")

# Dataset Configuration
datasets = {
    "CEDAR": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/CEDAR",
        "train_writers": list(range(261, 300)),
        "test_writers": list(range(300, 315))
    },
    "BHSig260_Bengali": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Bengali",
        "train_writers": list(range(1, 71)),
        "test_writers": list(range(71, 100))
    },
    "BHSig260_Hindi": {
        "path": "/Users/christelle/Downloads/Thesis/Dataset/BHSig260_Hindi",
        "train_writers": list(range(101, 169)),
        "test_writers": list(range(170, 260))
    },
}

# Apply SMOTE for Class Balancing
def apply_smote(X1, X2, y, sampling_strategy=1.0):
    flat_shape = (X1.shape[0], -1)
    X1_flat = X1.reshape(flat_shape)
    X2_flat = X2.reshape(flat_shape)

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    combined_resampled, y_resampled = smote.fit_resample(np.hstack([X1_flat, X2_flat]), y)

    X1_resampled = combined_resampled[:, :X1_flat.shape[1]].reshape(-1, *X1.shape[1:])
    X2_resampled = combined_resampled[:, X1_flat.shape[1]:].reshape(-1, *X2.shape[1:])

    print(f"SMOTE applied: {len(y_resampled) - len(y)} new samples added.")
    return X1_resampled, X2_resampled, y_resampled

# Training Process for Each Dataset
for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Training Model on {dataset_name} ---")

    # ✅ Create a new generator for each dataset
    generator = SignatureDataGenerator(
        dataset={dataset_name: dataset_config},
        img_height=155,
        img_width=220,
        batch_sz=4
    )

    (train_X1, train_X2), train_labels = generator.get_train_data()
    (test_X1, test_X2), test_labels = generator.get_test_data()
    train_labels = train_labels.astype(np.float32)  # You could also use np.int32
    test_labels = test_labels.astype(np.float32)  # You could also use np.int32

    if train_X1 is None or test_X1 is None:
        print(f"⚠ No data generated for {dataset_name}, skipping...")
        continue

    model = create_siamese_network(input_shape=(155, 220, 3))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss=triplet_loss)

    print(f"Train X1 shape: {train_X1.shape}, dtype: {train_X1.dtype}")
    print(f"Train X2 shape: {train_X2.shape}, dtype: {train_X2.dtype}")
    print(f"Train labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")

    start_time = time.perf_counter()
    model.fit(
        [train_X1, train_X2], train_labels, 
        epochs=5, 
        batch_size=4, 
        verbose=1
        )

    log_metrics(start_time, f"Training {dataset_name} Completed")

    model.save(f"{dataset_name}_siamese_model.h5")
    print(f"✅ Model saved as {dataset_name}_siamese_model.h5")
