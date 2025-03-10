import sys
import os
import numpy as np
import time
import faiss
import tensorflow as tf
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Flatten, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, ZeroPadding2D
from tensorflow.keras.optimizers import RMSprop
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from SignatureDataGenerator import SignatureDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

# Ensure reproducibility
import random
np.random.seed(1337)
random.seed(1337)

tf.config.set_soft_device_placement(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)

# Define the Euclidean distance function and Triplet Loss
def euclidean_distance(vectors):
    x, y = vectors
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    loss = tf.reduce_mean(tf.maximum(pos_dist - neg_dist + alpha, 0.0))
    return loss

# Define the base network
def create_base_network(input_shape):
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu')
    ])
    return model

def create_triplet_network(input_shape):
    base_network = create_base_network(input_shape)

    input_anchor = Input(shape=input_shape, name='input_anchor')
    input_positive = Input(shape=input_shape, name='input_positive')
    input_negative = Input(shape=input_shape, name='input_negative')

    encoded_anchor = base_network(input_anchor)
    encoded_positive = base_network(input_positive)
    encoded_negative = base_network(input_negative)

    distance = Lambda(euclidean_distance, output_shape=lambda x: x, name='distance')([encoded_anchor, encoded_positive, encoded_negative])

    model = Model(inputs=[input_anchor, input_positive, input_negative], outputs=distance)
    return model

# Build FAISS Index
def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# Search FAISS Index
def search_faiss(index, query_vectors, k=5):
    query_vectors = np.array(query_vectors, dtype=np.float32)
    distances, indices = index.search(query_vectors, k)
    return distances, indices

# Apply SMOTE for Class Balancing
def apply_smote(X1, X2, y, sampling_strategy=0.5):
    flat_shape = (X1.shape[0], -1)
    X1_flat = X1.reshape(flat_shape)
    X2_flat = X2.reshape(flat_shape)

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    combined = np.hstack([X1_flat, X2_flat])
    combined_resampled, y_resampled = smote.fit_resample(combined, y)

    X1_resampled = combined_resampled[:, :X1_flat.shape[1]].reshape(-1, *X1.shape[1:])
    X2_resampled = combined_resampled[:, X1_flat.shape[1]:].reshape(-1, *X2.shape[1:])

    return X1_resampled, X2_resampled, y_resampled

# Data Augmentation
def augment_data(X1, X2, y, augmentation_factor=2):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    X1_augmented, X2_augmented, y_augmented = [], [], []

    for i in range(len(X1)):
        for _ in range(augmentation_factor):
            X1_augmented.append(datagen.random_transform(X1[i]))
            X2_augmented.append(datagen.random_transform(X2[i]))
            y_augmented.append(y[i])

    return np.array(X1_augmented), np.array(X2_augmented), np.array(y_augmented)

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
    }
}

#Training Loop
for dataset_name, dataset_config in datasets.items():
    print(f"\n--- Training Model on {dataset_name} ---")

    generator = SignatureDataGenerator(
        dataset={dataset_name: dataset_config},
        img_height=155,
        img_width=220,
        batch_sz=4
    )

    train_data_generator = generator.get_train_data()

    train_X1, train_X2, train_labels = [], [], []

    for batch_X1, batch_X2, batch_labels in train_data_generator:
        train_X1.append(batch_X1)
        train_X2.append(batch_X2)
        train_labels.append(batch_labels)

    train_X1 = np.concatenate(train_X1, axis=0)
    train_X2 = np.concatenate(train_X2, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    train_X1, val_X1, train_X2, val_X2, train_labels, val_labels = train_test_split(
        train_X1, train_X2, train_labels, test_size=0.2, random_state=42
    )

    print(f"Train X1 shape: {train_X1.shape}, dtype: {train_X1.dtype}")
    print(f"Train X2 shape: {train_X2.shape}, dtype: {train_X2.dtype}")
    print(f"Train labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")
    train_labels = tf.cast(train_labels, tf.float32)

    train_X1, train_X2, train_labels = apply_smote(train_X1, train_X2, train_labels)

    print(f"Train X1 shape after SMOTE: {train_X1.shape}, dtype: {train_X1.dtype}")
    print(f"Train X2 shape after SMOTE: {train_X2.shape}, dtype: {train_X2.dtype}")
    print(f"Train labels shape after SMOTE: {train_labels.shape}, dtype: {train_labels.dtype}")

    train_X1, train_X2, train_labels = augment_data(train_X1, train_X2, train_labels)

    print(f"Train X1 shape after augmentation: {train_X1.shape}, dtype: {train_X1.dtype}")
    print(f"Train X2 shape after augmentation: {train_X2.shape}, dtype: {train_X2.dtype}")
    print(f"Train labels shape after augmentation: {train_labels.shape}, dtype: {train_labels.dtype}")

    input_shape = (155, 220, 3)
    triplet_model = create_triplet_network(input_shape)
    triplet_model.compile(optimizer=RMSprop(learning_rate=0.001), loss=triplet_loss)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=5,
        restore_best_weights=True
    )

    start_time = time.perf_counter()
    triplet_model.fit(
        [train_X1, train_X2, train_X2],  # Assuming train_X2 is used as both positive and negative for simplicity
        np.zeros((len(train_X1),)),  # Dummy labels
        epochs=10,
        batch_size=4,
        validation_data=([val_X1, val_X2, val_X2], np.zeros((len(val_X1),))),  # Validation data
        verbose=1,
        callbacks=[early_stopping]  # Include early stopping
    )

    log_metrics(start_time, f"Training {dataset_name} Completed")

    triplet_model.save(f"{dataset_name}_triplet_model.h5")
    print(f"✅ Model saved as {dataset_name}_triplet_model.h5")

    start_time = time.perf_counter()
    triplet_model.predict([train_X1[:1], train_X2[:1], train_X2[:1]])
    inference_time = time.perf_counter() - start_time
    print(f"Inference Time per Signature Pair: {inference_time:.4f}s")

    num_pairs = len(train_X1) * len(train_X2)
    print(f"Total Number of Pairwise Comparisons: {num_pairs}")

    embeddings = triplet_model.predict([train_X1, train_X2, train_X2])
    faiss_index = build_faiss_index(embeddings)
    start_time = time.perf_counter()
    distances, indices = search_faiss(faiss_index, embeddings[:1])
    faiss_query_time = time.perf_counter() - start_time
    print(f"FAISS Query Time: {faiss_query_time:.4f}s")

    memory_usage = (tf.config.experimental.get_memory_info('GPU:0')['peak'] / (1024 * 1024)
                    if tf.config.experimental.list_physical_devices('GPU') else 0)  # GPU memory
    print(f"Memory Usage: {memory_usage:.2f} MB")