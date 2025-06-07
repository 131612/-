import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from collections import Counter
import os
import matplotlib.pyplot as plt

# 导入数据
data_dir = "D:/Animal/raw-img"
img_size = (224, 224)
batch_size = 32

class_names = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']

# 数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 训练集
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# 验证集
val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 迁移学习，加载预训练的MobileNetV2
base_model = applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

# 添加自定义顶层分类器
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 计算类别权重
class_counts = Counter(train_generator.classes)
total_samples = sum(class_counts.values())
class_weights = {
    i: total_samples / (len(class_counts) * count)
    for i, count in class_counts.items()
}

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 定义回调
callbacks = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint(
        'animal_classifier.h5',
        save_best_only=True,
        monitor='val_accuracy'
    )
]

# 训练
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    epochs=20,
    callbacks=callbacks,
    class_weight=class_weights
)


# 加载最佳模型
model = models.load_model('animal_classifier.h5')

# 验证集评估
val_loss, val_acc = model.evaluate(val_generator)
print(f"\n验证集准确率: {val_acc:.4f}")

# 生成混淆矩阵
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 获取验证集真实标签和预测结果
val_preds = model.predict(val_generator)
val_pred_classes = np.argmax(val_preds, axis=1)
val_true_classes = val_generator.classes

# 混淆矩阵
cm = confusion_matrix(val_true_classes, val_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 分类报告
print(classification_report(
    val_true_classes, val_pred_classes,
    target_names=class_names, digits=4
))

#  结果可视化
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

model.save("animal_classifier_finetuned.keras")
print(" 模型保存完成：animal_classifier_finetuned.keras")