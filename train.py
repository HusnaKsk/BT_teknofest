import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import os
import shutil
import random
from tensorflow.keras.models import load_model
import cv2

"""
# Veri artırma ve normalizasyon
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 0-1 arasına ölçekle
    rotation_range=20,  # Rastgele döndürme
    width_shift_range=0.1,  # Genişlik kaydırma
    height_shift_range=0.1,  # Yükseklik kaydırma
    shear_range=0.1,  # Kesme dönüşümü
    zoom_range=0.1,  # Yakınlaştırma
    horizontal_flip=True  # Yatay çevirme
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Test seti için sadece normalizasyon

# Train verisini yükle
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Test verisini yükle
test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
    """
"""
)
# CNN modelini oluştur
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Overfitting'i azaltmak için
    layers.Dense(1, activation='sigmoid')  # Binary classification için
])

# Modeli derle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modelin özetini yazdır
model.summary()

# Modeli eğit
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=test_generator
)
model.save('bt_inme_model.h5')
print("✅ Model başarıyla kaydedildi: bt_inme_model.h5")


# Eğitim doğruluğu ve kaybı görselleştir
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

# Test verisini al
y_true = test_generator.classes  # Gerçek etiketler
y_pred = model.predict(test_generator) > 0.5  # Modelin tahmini

# F1 Score hesapla
f1 = f1_score(y_true, y_pred)
print(f"✅ F1 Score: {f1:.2f}")

"""

# Eğitilmiş modeli yükle
model = load_model('bt_inme_model.h5')
print("✅ Model başarıyla yüklendi ve kullanıma hazır!")


# Görüntüyü modele uygun hale getirme fonksiyonu
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Görüntüyü gri tonlamaya çevir
    img = cv2.resize(img, (224, 224))  # Modelin giriş boyutuna göre yeniden boyutlandır
    img = np.expand_dims(img, axis=-1)  # Kanal boyutu ekle (Gray -> (224, 224, 1))
    img = np.expand_dims(img, axis=0)  # Batch boyutu ekle (1, 224, 224, 1)
    img = img / 255.0  # Normalizasyon yap

    # Model tahmini yap
    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        return f"⚠️ İnme Var (Skor: {prediction:.2f})"
    else:
        return f"✅ İnme Yok (Skor: {prediction:.2f})"

# Kullanıcıdan bir PNG dosyası iste ve sonucu göster
image_path = "test_image.png"  # Buraya test etmek istediğin PNG dosyanın adını yaz
result = predict_image(image_path)
print(f"🧠 Modelin Tahmini: {result}")





