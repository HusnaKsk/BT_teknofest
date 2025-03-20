from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# **Veri artırma (Data Augmentation) ve ölçekleme**
train_image_datagen = ImageDataGenerator(rescale=1./255)
train_mask_datagen = ImageDataGenerator(rescale=1./255)

# **Görüntüleri yükle (X)**
train_image_generator = train_image_datagen.flow_from_directory(
    'bt/dataset2/images',  # **BT görüntülerinin olduğu klasör**
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,  # **Sınıf yok çünkü segmentasyon yapıyoruz**
    color_mode="rgb"  # **Görüntüler renkli olduğu için**
)

# **Maskeleri yükle (Y)**
train_mask_generator = train_mask_datagen.flow_from_directory(
    'bt/dataset2/masks',  # **Maskelerin olduğu klasör**
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,  # **Sınıf yok çünkü segmentasyon yapıyoruz**
    color_mode="grayscale"  # **Maskeler siyah-beyaz olduğu için**
)

# **Görüntüleri ve maskeleri eşleştir**
train_generator = zip(train_image_generator, train_mask_generator)

# **Modeli eğit**
history = unet_model.fit(
    train_generator,
    epochs=25,
    steps_per_epoch=len(train_image_generator),
    validation_data=train_generator,
    validation_steps=len(train_image_generator) // 5
)

# **Eğitilmiş modeli kaydet**
_model.save('unet_inme_model.h5')
print("✅ Model başarıyla kaydedildi!")
