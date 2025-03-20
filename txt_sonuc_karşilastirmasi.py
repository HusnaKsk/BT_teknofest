import tensorflow as tf
import numpy as np
import cv2
import os

# Eğitilmiş modeli yükle
model = tf.keras.models.load_model('bt_inme_model.h5')

# Test verilerinin olduğu klasör ve etiketlerin bulunduğu txt dosyası
test_folder = "dataset/veri_testseti"  # Test seti klasörü
labels_file = "ASAMA1_Cevaplar.txt"  # Gerçek etiketlerin olduğu dosya

# Eşik değeri (Threshold)
THRESHOLD = 0.7

# Etiketleri yükleme (Dosya formatı: "213131 0" veya "12345 1")
etiketler = {}
with open(labels_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()  # Boşluk ile ayrılmış
        if len(parts) == 2:
            dosya_numarasi, etiket = parts
            etiketler[dosya_numarasi] = int(float(etiket))  # Sayısal etiketi tam sayıya çevir

"""
# Görüntüyü modele uygun hale getirme fonksiyonu
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Görüntüyü renkli oku
    img = cv2.resize(img, (224, 224))  # Model giriş boyutuna göre yeniden boyutlandır
    img = img / 255.0  # Normalizasyon yap
    img = np.expand_dims(img, axis=0)  # Batch boyutu ekle (1, 224, 224, 3)

    # Model tahmini yap
    prediction = model.predict(img)[0][0]  # 0-1 arasında değer döndürür

    # İnme var/yok kararını ver (0 = Yok, 1 = Var)
    return 1 if prediction > THRESHOLD else 0

"""

# Model tahminini ters çeviren fonksiyon
def ters_predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    return 0 if prediction > THRESHOLD else 1  # Model tahminini ters çevir







# Yanlış tahmin edilenleri saklayacak liste
yanlis_tahminler = []
dogru_sayisi = 0
toplam_tahmin = 0

# Test klasöründeki tüm dosyaları işle
if not os.path.exists(test_folder):
    print("🚨 HATA: 'veri_testseti' klasörü bulunamadı! Lütfen dizini kontrol et.")
else:
    print("🧠 Model Test Setini İşliyor...\n")
    for file_name in os.listdir(test_folder):
        if file_name.endswith(".png"):  # Sadece PNG dosyalarını al
            dosya_numarasi = file_name.replace(".png", "")  # .png uzantısını kaldır
            image_path = os.path.join(test_folder, file_name)

            # Eğer test dosyasının etiketi varsa tahmin yap
            if dosya_numarasi in etiketler:
                gercek_etiket = etiketler[dosya_numarasi]  # Txt dosyasındaki gerçek değer
                model_tahmini = ters_predict_image(image_path)  # Model tahmini

                toplam_tahmin += 1
                if model_tahmini == gercek_etiket:
                    dogru_sayisi += 1
                else:
                    yanlis_tahminler.append((file_name, model_tahmini, gercek_etiket))

                print(f"📂 {file_name} → Tahmin: {model_tahmini}, Gerçek: {gercek_etiket}")

    # Model performansını göster
    print("\n📊 **Tahmin Özeti:**")
    print(f"✅ Doğru Tahmin Sayısı: {dogru_sayisi} / {toplam_tahmin}")
    print(f"❌ Yanlış Tahmin Sayısı: {len(yanlis_tahminler)}")

    # Yanlış tahmin edilenleri göster
    print("\n🚨 **Yanlış Tahmin Edilenler:**")
    for dosya, tahmin, gercek in yanlis_tahminler:
        print(f"❌ {dosya} → Model: {tahmin}, Doğru: {gercek}")

