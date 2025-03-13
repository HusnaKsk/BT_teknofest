import os

data_dir = "C:/PycharmProjeler/Teknofest_Bt/dataset/images/"

# Klasörleri ve içindekileri listele
if os.path.exists(data_dir):
    print("Ana klasör bulundu:", data_dir)
    print("Alt klasörler:", os.listdir(data_dir))  # images klasörünün içindekiler
    for subdir in os.listdir(data_dir):
        sub_path = os.path.join(data_dir, subdir)
        if os.path.isdir(sub_path):
            print(f"📂 {subdir} klasörü bulundu, içinde şu dosyalar var:")
            print(os.listdir(sub_path))  # Klasör içindeki dosyaları göster
else:
    print("HATA: Klasör yolu yanlış!")
