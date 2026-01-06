# ======================================================
# CNN OCR A-Z — KECIL & CEPAT
# ======================================================

import csv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ----------------------
# KONFIG (DIKECILIN)
# ----------------------
DATASET_PATH = "A_Z Handwritten Data.csv"
EPOCHS = 3
BATCH_SIZE = 32
MAX_DATA = 50000   # 🔥 BATAS DATA (KECIL & CEPAT)

# ----------------------
# LOAD CSV (BATAS DATA)
# ----------------------
X, y = [], []

with open(DATASET_PATH, newline='', encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for i, row in enumerate(reader):
        if i >= MAX_DATA:
            break
        y.append(int(row[0]) - 1)
        X.append([int(p)/255.0 for p in row[1:]])

print("Total data dipakai:", len(X))

# ----------------------
# DATASET TENSORFLOW
# ----------------------
X = tf.reshape(tf.convert_to_tensor(X, tf.float32), (-1, 28, 28, 1))
y = to_categorical(y, 26)

ds = tf.data.Dataset.from_tensor_slices((X, y))
ds = ds.shuffle(5000)

train_size = int(0.8 * len(X))
train_ds = ds.take(train_size).batch(BATCH_SIZE)
test_ds  = ds.skip(train_size).batch(BATCH_SIZE)

# ----------------------
# MODEL (LEBIH RINGAN)
# ----------------------
model = Sequential([
    Conv2D(16, 3, activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(26, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------
# TRAIN
# ----------------------
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    verbose=1
)

# ----------------------
# SAVE
# ----------------------
model.save("cnn_ocr_az_small.h5")

# ----------------------
# PLOT
# ----------------------
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["Train", "Val"])
plt.title("Accuracy")
plt.show()
