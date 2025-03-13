import os

dataset_path = r"C:\PycharmProjeler\Teknofest_Bt\dataset\images"

# Klasörleri listele
class_names = os.listdir(dataset_path)

print("📌 Veri setindeki mevcut sınıflar:")
for class_name in class_names:
    print(f"➡ {class_name}")
