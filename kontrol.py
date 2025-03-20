import tensorflow as tf
import numpy as np
import cv2

# Eğitilmiş modeli yükle
model = tf.keras.models.load_model('bt_inme_model.h5')

# Görüntüyü modele uygun hale getirme fonksiyonu
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Görüntüyü renkli olarak oku
    img = cv2.resize(img, (224, 224))  # Modelin giriş boyutuna göre yeniden boyutlandır
    img = img / 255.0  # Normalizasyon yap
    img = np.expand_dims(img, axis=0)  # Batch boyutu ekle (1, 224, 224, 3)

    # Model tahmini yap
    prediction = model.predict(img)[0][0]
    if prediction > 0.8:
        return f"⚠️ İnme Var (Skor: {prediction:.2f})"
    else:
        return f"✅ İnme Yok (Skor: {prediction:.2f})"

# Kullanıcıdan bir PNG dosyası iste ve sonucu göster
image_path = "17026.png"  # Buraya test etmek istediğin PNG dosyanın adını yaz
result = predict_image(image_path)
print(f"🧠 Modelin Tahmini: {result}")



