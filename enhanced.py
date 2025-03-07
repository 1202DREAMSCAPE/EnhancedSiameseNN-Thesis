import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import faiss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.optimizers.legacy import RMSprop
from tensorflow.keras import backend as K
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from SignatureDataGenerator import SignatureDataGenerator
from SigNet_v1 import create_siamese_network
from tensorflow.keras.models import load_model
from RealESRGAN.realesrgan.utils import RealESRGANer

# Ensure reproducibility
import random
np.random.seed(1337)
random.seed(1337)


# Initialize Real-ESRGAN Model
model_path = "RealESRGAN/weights/RealESRGAN_x4plus.pth"
try:
    esrgan = RealESRGANer(
        scale=4,
        model_path=model_path,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("✅ Real-ESRGAN model loaded successfully!")
except Exception as e:
    print("❌ Error loading Real-ESRGAN model:", e)
    esrgan = None

# Enable GPU & Prevent Out-of-Memory (OOM) Errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)

# Mixed Precision for Faster Training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Triplet Loss Function
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = K.sum(K.square(anchor - positive), axis=-1)
    neg_dist = K.sum(K.square(anchor - negative), axis=-1)
    loss = K.maximum(pos_dist - neg_dist + alpha, 0.0)
    return K.mean(loss)

# FAISS for efficient similarity search
def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def search_faiss(index, query_vectors, k=5):
    query_vectors = np.array(query_vectors, dtype=np.float32)
    distances, indices = index.search(query_vectors, k)
    return distances, indices

# Apply FAISS for Efficient Embedding Search
def get_hard_negatives(embeddings, labels, k=5):
    index = build_faiss_index(embeddings)
    distances, indices = search_faiss(index, embeddings, k)
    hard_negatives = [(i, j) for i in range(len(labels)) for j in indices[i] if labels[i] != labels[j]]
    return hard_negatives

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

# Load Dataset with Real-ESRGAN and SMOTE
def load_data_with_esrgan_and_smote(dataset_name, dataset_config):
    generator = SignatureDataGenerator(
        dataset={dataset_name: dataset_config},
        img_height=155, img_width=220,
        apply_esrgan=True
    )
    generator.esrgan_model = esrgan  # Set Real-ESRGAN model

    (train_X1, train_X2), train_labels = generator.get_train_data()
    (test_X1, test_X2), test_labels = generator.get_test_data()

    # Apply SMOTE after Real-ESRGAN
    train_X1, train_X2, train_labels = apply_smote(train_X1, train_X2, train_labels)

    return (train_X1, train_X2, train_labels), (test_X1, test_X2, test_labels)

# Train & Save Separate Models
for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Training Model on {dataset_name} ---")

    (train_X1, train_X2, train_labels), (test_X1, test_X2, test_labels) = load_data_with_esrgan_and_smote(dataset_name, dataset_config)

    model = create_siamese_network(input_shape=(155, 220, 1))
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss=triplet_loss)

    model.fit((train_X1, train_X2), train_labels, epochs=5, batch_size=8, verbose=1)
    model.save(f"{dataset_name}_siamese_model.h5")
    print(f"Model saved as {dataset_name}_siamese_model.h5")

# Train Unified Model on All Datasets
unified_model = create_siamese_network(input_shape=(155, 220, 1))
unified_model.compile(optimizer=RMSprop(learning_rate=0.001), loss=triplet_loss)
unified_model.fit((train_X1, train_X2), train_labels, epochs=5, batch_size=8, verbose=1)
unified_model.save("unified_siamese_model.h5")
print("Unified model saved as unified_siamese_model.h5")