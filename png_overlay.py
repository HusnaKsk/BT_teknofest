import cv2
import os
import numpy as np

# Klasör yollarını tam yol olarak ayarla
base_folder = r"C:\PycharmProjeler\bt\dataset2"
image_folder = os.path.join(base_folder, "images")
overlay_folder = os.path.join(base_folder, "overlay")
mask_output_folder = os.path.join(base_folder, "masks")

# Eğer mask klasörleri yoksa oluştur
os.makedirs(os.path.join(mask_output_folder, "iskemi"), exist_ok=True)
os.makedirs(os.path.join(mask_output_folder, "kanama"), exist_ok=True)

# İşlenecek kategori isimleri (normal görüntüler overlay olmadığı için hariç tutuluyor)
categories = ["iskemi", "kanama"]

for category in categories:
    image_category_path = os.path.join(image_folder, category)
    overlay_category_path = os.path.join(overlay_folder, category)
    mask_category_path = os.path.join(mask_output_folder, category)

    # Eğer klasör eksikse hata vermesin
    if not os.path.exists(image_category_path):
        print(f"🚨 HATA: {image_category_path} klasörü bulunamadı! Devam ediliyor...")
        continue
    if not os.path.exists(overlay_category_path):
        print(f"🚨 HATA: {overlay_category_path} klasörü bulunamadı! Devam ediliyor...")
        continue

    for file_name in os.listdir(image_category_path):
        if file_name.endswith(".png"):  # Sadece PNG dosyalarını işle
            image_path = os.path.join(image_category_path, file_name)
            overlay_path = os.path.join(overlay_category_path, file_name)  # Aynı isimli overlay dosyası aranacak

            # Eğer overlay dosyası varsa işle
            if os.path.exists(overlay_path):
                # Görüntüleri yükle
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Ana görüntü (grayscale)
                overlay = cv2.imread(overlay_path, cv2.IMREAD_GRAYSCALE)  # Overlay görüntüsü (grayscale)

                # Overlay'ı çıkar ve maske oluştur
                mask = cv2.absdiff(image, overlay)  # Farkları al
                _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)  # Maske oluştur

                # Yeni maskeyi kaydet
                mask_output_path = os.path.join(mask_category_path, file_name)
                cv2.imwrite(mask_output_path, mask)

                print(f"✅ Maske kaydedildi: {mask_output_path}")
            else:
                print(f"⚠️ Uyarı: {file_name} için overlay bulunamadı, atlanıyor.")
