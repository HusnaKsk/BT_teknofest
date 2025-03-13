import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Veri setinin yolu
dataset_path = r"C:\PycharmProjeler\Teknofest_Bt\dataset"

# Maskelerin olduğu klasörler
mask_folders = {
    "iskemi": os.path.join(dataset_path, "masks", "iskemi"),
    "kanama": os.path.join(dataset_path, "masks", "kanama")
}

# Overlay dosyalarının olduğu klasörler
overlay_folders = {
    "iskemi": os.path.join(dataset_path, "overlay", "iskemi"),
    "kanama": os.path.join(dataset_path, "overlay", "kanama")
}

# İşaretleme oranını hesaplayan fonksiyon
def get_mask_ratio(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Maskeyi yükle
    if mask is None:
        return 0  # Eğer maske yüklenemiyorsa hata verme
    total_pixels = mask.size  # Toplam piksel sayısı
    labeled_pixels = np.count_nonzero(mask)  # İşaretlenmiş piksel sayısı
    return (labeled_pixels / total_pixels) * 100  # Yüzdelik oran

# Küçük işaretlemeleri tespit et
small_masks = []

for category, mask_folder in mask_folders.items():
    if os.path.exists(mask_folder):
        for mask_file in os.listdir(mask_folder):
            mask_path = os.path.join(mask_folder, mask_file)
            mask_ratio = get_mask_ratio(mask_path)
            if mask_ratio < 1:  # %1'den küçük işaretleme varsa
                small_masks.append((category, mask_file, mask_ratio))

# Eğer küçük işaretlemeler varsa listeleyelim
if small_masks:
    print(f"Toplam {len(small_masks)} maske çok küçük işaretlemeler içeriyor!")
    for category, mask_file, ratio in small_masks[:10]:  # İlk 10 tanesini gösterelim
        print(f"{category} - {mask_file} | İşaretleme Oranı: {ratio:.4f}%")
else:
    print("Tüm maskelerde yeterli işaretleme var, küçük işaretlemeler yok.")

# Küçük işaretlemeleri overlay ile görselleştirelim
num_samples = min(5, len(small_masks))  # En fazla 5 örnek gösterelim

if num_samples > 0:
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
    fig.suptitle("Küçük İşaretlemeli Maskeler ve Overlay Görselleri")

    for i, (category, mask_file, ratio) in enumerate(small_masks[:num_samples]):
        mask_path = os.path.join(mask_folders[category], mask_file)
        overlay_path = os.path.join(overlay_folders[category], mask_file)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        overlay = cv2.imread(overlay_path)

        if num_samples == 1:
            axes[0].imshow(mask, cmap='gray')
            axes[0].set_title(f"Küçük İşaretleme ({ratio:.4f}%)")
            axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[1].set_title("Overlay Görüntüsü")
        else:
            axes[i, 0].imshow(mask, cmap='gray')
            axes[i, 0].set_title(f"Küçük İşaretleme ({ratio:.4f}%)")
            axes[i, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[i, 1].set_title("Overlay Görüntüsü")

    plt.show()

else:
    print("Küçük işaretlemeler bulundu ancak overlay görüntülerini göstermek için yeterli örnek yok.")
