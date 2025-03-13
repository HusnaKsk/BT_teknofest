import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 📂 Veri setinin yolu
dataset_path = r"C:\PycharmProjeler\Teknofest_Bt\dataset\images"

# ResNet için uygun giriş boyutu
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ✅ Veri artırma (augmentation) işlemi
datagen = ImageDataGenerator(
    rescale=1./255,         # Normalizasyon (0-1 aralığına çekiyoruz)
    rotation_range=10,      # Hafif döndürme
    width_shift_range=0.1,  # Yatay kaydırma
    height_shift_range=0.1, # Dikey kaydırma
    horizontal_flip=True,   # Yatay çevirme
    validation_split=0.2    # %20 validation set olarak ayır
)

# 🏆 Eğitim verisi (Train Set)
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    classes=["iskemi", "kanama", "normal"]  # Sadece bu 3 sınıfı kullan!
)

# 🏆 Doğrulama verisi (Validation Set)
val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    classes=["iskemi", "kanama", "normal"]
)
