import os

inme_var_path = "dataset/test/inme_var"
inme_yok_path = "dataset/test/inme_yok"

print(f"🧠 İnme Var Görselleri: {len(os.listdir(inme_var_path))}")
print(f"✅ İnme Yok Görselleri: {len(os.listdir(inme_yok_path))}")
