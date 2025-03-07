import time
import numpy as np
import psutil
import tensorflow as tf
import faiss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

# Load the saved model
model_path = "unified_siamese_model.h5"
model = load_model(model_path, compile=False)
print(f"Model loaded from {model_path}")

# Simulated test data (Replace with actual dataset loading)
num_samples = 1000
embedding_dim = 128
np.random.seed(42)
test_X1 = np.random.rand(num_samples, 155, 220, 1).astype(np.float32)
test_X2 = np.random.rand(num_samples, 155, 220, 1).astype(np.float32)
test_labels = np.random.randint(0, 2, num_samples)  # Genuine(1) or Forged(0)

# Convert test data into TensorFlow dataset
test_dataset = tf.data.Dataset.from_tensor_slices(((test_X1, test_X2), test_labels)).batch(16)

# Measure inference time
start_time = time.time()
y_pred = model.predict(test_dataset)
inference_time = (time.time() - start_time) / num_samples

# Compute FAISS query time
faiss_index = faiss.IndexFlatL2(embedding_dim)
faiss_index.add(np.random.rand(num_samples, embedding_dim).astype(np.float32))
query_vectors = np.random.rand(10, embedding_dim).astype(np.float32)
start_time = time.time()
_, _ = faiss_index.search(query_vectors, 5)
faiss_query_time = (time.time() - start_time) / 10

# Compute classification metrics
y_pred_labels = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(test_labels, y_pred_labels)
precision = precision_score(test_labels, y_pred_labels, average='binary')
recall = recall_score(test_labels, y_pred_labels, average='binary')
f1 = f1_score(test_labels, y_pred_labels, average='binary')

# Measure memory usage
memory_used = psutil.Process().memory_info().rss / (1024 * 1024)  # In MB

# Display metrics
print("\n--- Model Evaluation Metrics ---")
print(f"Inference Time per Sample: {inference_time:.6f} sec")
print(f"FAISS Query Time: {faiss_query_time * 1000:.4f} ms")
print(f"Memory Usage: {memory_used:.2f} MB")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
