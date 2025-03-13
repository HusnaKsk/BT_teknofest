import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Veri seti yolu
data_dir = "Teknofest_bt/dataset/images"

# Veri ön işleme
img_size = (224, 224)
batch_size = 32

# Data Augmentation ve Normalizasyon
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # %20 validation
)

# Eğitim ve Doğrulama Verileri
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# CNN Modeli
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 sınıf olduğu için
])

# Model Derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli Eğitme
epochs = 10
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

# Modeli Kaydetme
model.save("stroke_detection_model.h5")
print("Model kaydedildi: stroke_detection_model.h5")
