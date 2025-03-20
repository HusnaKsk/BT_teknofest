import tensorflow as tf
import numpy as np
import cv2
import os

# Eğitilmiş modeli yükle
model = tf.keras.models.load_model('bt_inme_model.h5')

# Test verilerinin olduğu klasör
test_folder = "dataset/veri_testseti"  # Buraya test seti klasörünü yaz

# Eşik değeri (Threshold), 0.5 veya 0.7 kullanabilirsin
THRESHOLD = 0.7

# Görüntüyü modele uygun hale getirme fonksiyonu
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Görüntüyü renkli olarak oku
    img = cv2.resize(img, (224, 224))  # Modelin giriş boyutuna göre yeniden boyutlandır
    img = img / 255.0  # Normalizasyon yap
    img = np.expand_dims(img, axis=0)  # Batch boyutu ekle (1, 224, 224, 3)

    # Model tahmini yap
    prediction = model.predict(img)[0][0]  # 0-1 arasında bir değer döndürür

    # Tahmini yazdır
    if prediction > THRESHOLD:
        return f"⚠️ İnme Var (Skor: {prediction:.2f})"
    else:
        return f"✅ İnme Yok (Skor: {prediction:.2f})"

# Test klasöründeki tüm dosyaları işle
if not os.path.exists(test_folder):
    print("🚨 HATA: 'veri_testseti' klasörü bulunamadı! Lütfen dizini kontrol et.")
else:
    print("🧠 Model Test Setini İşliyor...\n")
    for file_name in os.listdir(test_folder):
        if file_name.endswith(".png"):  # Sadece PNG dosyalarını al
            image_path = os.path.join(test_folder, file_name)
            result = predict_image(image_path)
            print(f"📂 Dosya: {file_name} → {result}")
