import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import keras 
import random
from  keras.applications.efficientnet import EfficientNetB4,preprocess_input,decode_predictions
import warnings 
warnings.simplefilter(action="ignore")

ICE_PATH = "/home/hice1/kleung43/scratch/cs7643"
train_path = ICE_PATH + "/data/seg_train/seg_train"
test_path =  ICE_PATH + "/data/seg_test/seg_test"
pred_path =  ICE_PATH + "/data/seg_pred/seg_pred"
img_size = (150,150)
mode="rgb"
batch_s = 128

# preprocessing
train_val_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.25,
    rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	fill_mode="nearest")

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_data = train_val_generator.flow_from_directory(
    train_path,
    target_size=img_size,
    color_mode=mode,
    batch_size=batch_s,
    class_mode="categorical",
    shuffle=True,
    seed=7,
    subset="training"
)

val_data = train_val_generator.flow_from_directory(
    train_path,
    target_size=img_size,
    color_mode=mode,
    batch_size=batch_s,
    class_mode="categorical",
    shuffle=True,
    seed=7,
    subset="validation"
)

test_data = test_generator.flow_from_directory(
    test_path,
    target_size=img_size,
    color_mode=mode,
    batch_size=batch_s,
    class_mode="categorical",
    shuffle=False,
    seed=7,
)

input_tensor = keras.layers.Input(shape=(150, 150, 3))
base_model = EfficientNetB4(include_top= False , weights= 'imagenet' , input_tensor = input_tensor , pooling= 'max')

x = base_model.output
X = keras.layers.Dropout(0.4)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dense(256,activation="relu")(x)
x = keras.layers.Dense(256,activation="relu")(x)
X = keras.layers.Dropout(0.5)(x)
predictions = keras.layers.Dense(6, activation='softmax')(x)

model = keras.models.Model(inputs=base_model.input, outputs=predictions)
   
model.compile(optimizer = keras.optimizers.Adamax(learning_rate = 0.0001), loss="categorical_crossentropy",metrics=["accuracy"])

history = model.fit(train_data,validation_data=val_data, epochs=25)

model.evaluate(test_data)

train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = [i+1 for i in range(len(train_acc))]

plt.figure(figsize=(12,9))
plt.subplot(2,1,1)
plt.plot(epochs,train_loss,'b',label="Train Loss")
plt.plot(epochs,val_loss,'g',label="Validation loss")
plt.title("Loss")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.figure(figsize=(12,9))
plt.subplot(2,1,1)
plt.plot(epochs,train_acc,'b',label="Train Accuracy")
plt.plot(epochs,val_acc,'g',label="Validation Accuracy")
plt.title("Accuracy")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

test_data.class_indices

pred_list = os.listdir(ICE_PATH + "/data/seg_pred/seg_pred")

for i in range(random.randint(30,40), random.randint(50,60)):
    path = os.path.join(pred_path, pred_list[i])
    img = cv2.imread(path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plt.show()
    img = np.expand_dims(img/255, 0)
    pred = np.argmax(model.predict(img))
    if pred == 0:
        print('Building')
    elif pred == 1:
        print('Forest')
    elif pred == 2:
        print('Glacier')
    elif pred == 3:
        print('Mountain')
    elif pred == 4:
        print('Sea')
    elif pred == 5:
        print('street')

predictions = model.predict(test_data) 
y_pred = np.argmax(predictions, axis = 1)
y_true = test_data.classes
print(classification_report(y_true,y_pred))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title("EfficientNetB0 confusion matrix")
plt.show()

model.save("intel_image_EfficientNetB4_model.keras")