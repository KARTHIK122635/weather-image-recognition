import os
import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pos  # make sure pos.py is in the same folder

# 1. Load images and labels
def load_dataset(data_path, image_size=(32, 32)):
    categories = ['cloud_1', 'shine_1', 'sunrise_1', 'rain_1']
    label_map = {'cloud_1': 0, 'shine_1': 1, 'sunrise_1': 2, 'rain_1': 3}
    
    X = []
    y = []
    
    for category in categories:
        files = glob.glob(os.path.join(data_path, category, '*.*'))
        for f in files:
            img = cv2.imread(f)
            img = cv2.resize(img, image_size)
            X.append(img)
            y.append(label_map[category])
    
    X = np.array(X) / 255.0  # Normalize
    y = to_categorical(np.array(y), num_classes=4)  # One-hot
    return X, y

# 2. Prepare dataset
X, y = load_dataset("weather_agu")  # Adjust path if needed
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# 3. Load model
model = pos.create_cct_model()

# 4. Compile model
optimizer = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001)
model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
    metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")]
)

# 5. Setup checkpoint callback
checkpoint_path = "checkpoint.weights.h5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_accuracy',
    verbose=1
)

# 6. Train model
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint_callback]
)

# 7. Final evaluation
model.load_weights(checkpoint_path)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")
