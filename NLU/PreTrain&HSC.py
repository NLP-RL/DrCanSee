# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive')


import matplotlib.pyplot as plt
import os
import tensorflow as tf
import matplotlib
matplotlib.style.use('ggplot')

IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = 'drive/MyDrive/final_ee/train'
VALID_DATA_DIR = 'drive/MyDrive/final_ee/test'

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
    # zoom_range=.2,
    # horizontal_flip=True,
    # rotation_range=30,
    # shear_range = 10,
    # width_shift_range=.2,
    # height_shift_range=.2
)
train_generator = datagen.flow_from_directory(
    TRAINING_DATA_DIR,
    shuffle=True,
    target_size=IMAGE_SHAPE,

)
valid_generator=datagen.flow_from_directory(TRAINING_DATA_DIR,target_size=(224,224),subset='validation',batch_size = 32)

test_generator = datagen.flow_from_directory(
    VALID_DATA_DIR,
    shuffle=False,
    target_size=IMAGE_SHAPE,
    batch_size = 1
)

from keras.models import Model
from tensorflow import keras
a = keras.models.load_model('drive/MyDrive/final_ee/a_model.h5')
b1 = keras.models.load_model('drive/MyDrive/final_ee/b1_model.h5')
b2 = keras.models.load_model('drive/MyDrive/final_ee/b2_model.h5')
b3 = keras.models.load_model('drive/MyDrive/final_ee/b3_model.h5')
b4 = keras.models.load_model('drive/MyDrive/final_ee/b4_model.h5')

indices  = train_generator.class_indices
print(indices)
b1_indices = {0: 'S108', 1: 'S181', 2: 'S22', 3: 'S51'}
b2_indices = {0: 'M97', 1: 'S1', 2: 'S37', 3: 'S5', 4: 'S74'}
b3_indices = {0: 'M61', 1: 'S11', 2: 'S177', 3: 'S251', 4: 'S31', 5:'S4'}
b4_indices = {0: 'M23', 1: 'S159', 2: 'S205', 3: 'S39', 4: 'S50'}

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import math as m

y_final = []
y_true = []
for i,(x_batch_val, y_batch_val) in enumerate(test_generator):
  print(i)
  # print(y_batch_val.shape)
  lis1 = y_batch_val.tolist()
  # print(lis1)
  y_true = y_true + [np.argmax(lis1[0])]
  logits = a.predict(x_batch_val)
  input = np.argmax(logits)
  if input==0:
    y_pred1 = b1.predict(x_batch_val)
    y1_max = np.argmax(y_pred1)
    s1 = b1_indices[y1_max]
    y_final.append(indices[s1])
  elif input==1:
    y_pred2 = b2.predict(x_batch_val)
    y2_max = np.argmax(y_pred2)
    s2 = b2_indices[y2_max]
    y_final.append(indices[s2])
  elif input==2:
    y_pred4 = b4.predict(x_batch_val)
    y4_max = np.argmax(y_pred4)
    s4 = b4_indices[y4_max]
    y_final.append(indices[s4])
  elif input==3:
    y_pred3 = b3.predict(x_batch_val)
    y3_max = np.argmax(y_pred3)
    s3 = b3_indices[y3_max]
    y_final.append(indices[s3])
  else:
    print('it is else')

  if m.ceil(test_generator.samples/1)==i+1:
    break
print(y_final)
print(len(y_final))
print(y_true)
print(len(y_true))
# print(predictions_valid)
# y_pred = []
# for i in range(0,len(predictions_valid)):
#     y_pred = y_pred + [np.argmax(predictions_valid[i])]
# print('predictions : ',y_pred)

x = test_generator.class_indices
k=[]
for i in x:
  if i not in k:
    k.append(i)
print(k)

score = accuracy_score(y_true, y_final)
print('Classification Report: ',classification_report(y_true,y_final))
cm = confusion_matrix(y_true, y_final)
f,ax= plt.subplots(figsize=(15, 15))
sns.heatmap(cm, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
classes_rev = []

classes = k
l = len(k)
for i in range(1,l+1):
    classes_rev = classes_rev + [classes[l-i]]
ax.xaxis.set_ticklabels(classes);ax.yaxis.set_ticklabels(classes);
ax.figure.savefig("hybrid_cnn.png")
print('Accuracy: ', score)
print('F1',sklearn.metrics.f1_score(y_true, y_final, average='weighted'))

n = len(y_true)

for i in range(n-1):
  for j in range(0, n-i-1):
    if y_true[j] > y_true[j + 1] :
      y_true[j], y_true[j + 1] = y_true[j + 1], y_true[j]
      y_final[j], y_final[j + 1] = y_final[j + 1], y_final[j]

print(y_true)
print(y_final)

y_z = []
temp = []
for i in range(0,200):
  temp.append(y_final[i])
  if (i+1)%10==0:
    y_z.append(temp)
    temp = []
print(y_z)
final = []
final_temp = []
for i in y_z:
  for j in i:
    final_temp.append(k[j])
  final.append(final_temp)
  final_temp = []
print(final)

dic = {}
c = 0
for i in final:
  dic[k[c]] = i
  c = c+1
print(dic)

import pickle

multimodal_dict ={'M23':'Edema', 'M61':'Cyanosis', 'M97':'Proptosis', 'S1':'Swollen eye', 'S108': 'Mouth ulcer', 'S11': 'Skin dryness, peeling, scaliness, or roughness', 'S159': 'Neck swelling', 'S177': 'Skin irritation', 'S181': 'Swollen or red tonsils', 'S205': 'Knee swelling', 'S22': 'Redness in ear', 'S251': 'Dry or flaky scalp', 'S31': 'Skin rash', 'S37': 'Itchy eyelid', 'S39': 'Hand or Finger lump or mass', 'S4':  'Skin growth', 'S5': 'Eye redness', 'S50': 'Foot or toe swelling', 'S51': 'Lip Swelling', 'S74': 'Eyelid lesion or rash'}
print(len(multimodal_dict))
multimodal_symp = ['Swollen eye','Skin growth','Eye redness','Skin dryness, peeling, scaliness, or roughness','Redness in ear','Skin rash','Itchy eyelid','Hand or Finger lump or mass','Foot or toe swelling','Lip Swelling','Eyelid lesion or rash','Mouth ulcer','Neck swelling','Skin irritation','Swollen or red tonsils','Knee swelling','Dry or flaky scalp']
# print(len(multimodal_symp))
P = {}
for i in dic:
  P[multimodal_dict[i]] = dic[i]
print(P)


new_sd = {}
ntemp = []
#SD
for i in P:
  if i!='Edema' and i!='Cyanosis' and i!='Proptosis':
    temp = P[i]
    for j in temp:
      ntemp.append(multimodal_dict[j])
    new_sd[i] = ntemp
    ntemp = []
print(len(new_sd))
print(new_sd)

pickle.dump(new_sd, open('SD_DesneNet169CNN_Hierarchical.p', 'wb'))

new_mdd = {}
ntemp = []
for i in P:
  if i=='Edema' or i=='Cyanosis' or i=='Proptosis':
    temp = P[i]
    for j in temp:
      ntemp.append(multimodal_dict[j])
    new_mdd[i] = ntemp
    ntemp = []
new_mdd['Ulcer'] = new_sd['Mouth ulcer']
new_mdd['Eye swelling'] = new_sd['Swollen eye']
print(len(new_mdd))
print(new_mdd)
pickle.dump(new_mdd, open('MDD_DesneNet169CNN_Hierarchical.p', 'wb'))











from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet152
from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
model = DenseNet169(include_top = False,input_shape=(224,224,3),weights = 'imagenet')
model.trainable = False

conv1 = tf.keras.layers.Conv2D(1024,3,1,'valid',activation='relu')(model.layers[-1].output)
conv2 = tf.keras.layers.Conv2D(512,2,1,'valid',activation='relu')(conv1)
gmax = tf.keras.layers.GlobalMaxPooling2D()(conv2)
# gmax = tf.keras.layers.GlobalMaxPooling2D()(model.layers[-1].output)
flat1 = Flatten()(gmax)
class1 = Dense(512, activation='relu')(flat1)
output = Dense(5, activation='softmax')(class1)
model = Model(inputs=model.inputs, outputs=output)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
# print(model.summary())

from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# print(train_generator.classes)
cls = compute_class_weight('balanced', classes = np.unique(train_generator.classes), y = train_generator.classes)
weight = {i : cls[i] for i in range(5)}
EPOCHS = 5
BATCH_SIZE = 32
history = model.fit(train_generator,
                    class_weight=weight,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=valid_generator,
                    validation_steps= valid_generator.samples // BATCH_SIZE,
                    verbose=1,
                    )

model.evaluate(test_generator)

model = keras.models.load_model('drive/MyDrive/final_ee/densenetcnn.h5')

# d =  ['M23','M61','M97', 'S1','S108','S11','S159','S177','S181','S205', 'S22','S251','S31','S37','S39', 'S4','S5','S50','S51','S74']
x = test_generator.class_indices
k=[]
for i in x:
  if i not in k:
    k.append(i)
print(k)
# print(x)

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns
predictions_valid = model.predict(test_generator)
# print(predictions_valid)
y_pred = []

for i in range(0,len(predictions_valid)):
    y_pred = y_pred + [np.argmax(predictions_valid[i])]

print('predictions : ',y_pred)
y = []
temp = []
for i in range(0,200):
  temp.append(y_pred[i])
  if (i+1)%10==0:
    y.append(temp)
    temp = []
print(y)
final = []
final_temp = []
for i in y:
  for j in i:
    final_temp.append(k[j])
  final.append(final_temp)
  final_temp = []
print(final)

y_true = test_generator.classes
print('true classes:',y_true)

score = accuracy_score(y_true, y_pred)
print('Classification Report: ',classification_report(y_true,y_pred))
cm = confusion_matrix(y_true, y_pred)
f,ax= plt.subplots(figsize=(15, 15))
sns.heatmap(cm, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
classes_rev = []

classes = k
l = len(k)
for i in range(1,l+1):
    classes_rev = classes_rev + [classes[l-i]]
ax.xaxis.set_ticklabels(classes);ax.yaxis.set_ticklabels(classes);
ax.figure.savefig("hybrid_cnn.png")
print('Accuracy: ', score)
print('F1',sklearn.metrics.f1_score(y_true, y_pred, average='weighted'))

final[2]
dic = {}
c = 0
for i in final:
  dic[k[c]] = i
  c = c+1
print(dic)

import pickle

multimodal_dict ={'M23':'Edema', 'M61':'Cyanosis', 'M97':'Proptosis', 'S1':'Swollen eye', 'S108': 'Mouth ulcer', 'S11': 'Skin dryness, peeling, scaliness, or roughness', 'S159': 'Neck swelling', 'S177': 'Skin irritation', 'S181': 'Swollen or red tonsils', 'S205': 'Knee swelling', 'S22': 'Redness in ear', 'S251': 'Dry or flaky scalp', 'S31': 'Skin rash', 'S37': 'Itchy eyelid', 'S39': 'Hand or Finger lump or mass', 'S4':  'Skin growth', 'S5': 'Eye redness', 'S50': 'Foot or toe swelling', 'S51': 'Lip Swelling', 'S74': 'Eyelid lesion or rash'}
print(len(multimodal_dict))
multimodal_symp = ['Swollen eye','Skin growth','Eye redness','Skin dryness, peeling, scaliness, or roughness','Redness in ear','Skin rash','Itchy eyelid','Hand or Finger lump or mass','Foot or toe swelling','Lip Swelling','Eyelid lesion or rash','Mouth ulcer','Neck swelling','Skin irritation','Swollen or red tonsils','Knee swelling','Dry or flaky scalp']
# print(len(multimodal_symp))
P = {}
for i in dic:
  P[multimodal_dict[i]] = dic[i]
print(P)


new_sd = {}
ntemp = []
#SD
for i in P:
  if i!='Edema' and i!='Cyanosis' and i!='Proptosis':
    temp = P[i]
    for j in temp:
      ntemp.append(multimodal_dict[j])
    new_sd[i] = ntemp
    ntemp = []
print(len(new_sd))
print(new_sd)

pickle.dump(new_sd, open('SD_DesneNet169CNN.p', 'wb'))

new_mdd = {}
ntemp = []
for i in P:
  if i=='Edema' or i=='Cyanosis' or i=='Proptosis':
    temp = P[i]
    for j in temp:
      ntemp.append(multimodal_dict[j])
    new_mdd[i] = ntemp
    ntemp = []
new_mdd['Ulcer'] = new_sd['Mouth ulcer']
new_mdd['Eye swelling'] = new_sd['Swollen eye']
print(len(new_mdd))
print(new_mdd)
pickle.dump(new_mdd, open('MDD_DesneNet169CNN.p', 'wb'))



def build_model(num_classes):
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
                           input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
model = build_model(num_classes=20)
print(model)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
print(model.summary())

EPOCHS = 30
BATCH_SIZE = 32
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=valid_generator,
                    validation_steps= valid_generator.samples // BATCH_SIZE,
                    verbose=1
                    )

model.evaluate(test_generator)

model.save('baseline_cnn.h5')
x = test_generator.class_indices
print(x)
d = ['M23','M61','M97', 'S1','S108','S11','S159','S177','S181','S205', 'S22','S251','S31','S37','S39', 'S4','S5','S50','S51','S74']

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns
predictions_valid = model.predict(test_generator)

y_pred = []

for i in range(0,len(predictions_valid)):
    y_pred = y_pred + [np.argmax(predictions_valid[i])]

print('predictions : ',y_pred)
y_true = test_generator.classes
print('true classes:',y_true)

score = accuracy_score(y_true, y_pred)
print('Classification Report: ',classification_report(y_true,y_pred))
cm = confusion_matrix(y_true, y_pred)
f,ax= plt.subplots(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
classes_rev = []
classes =  ['M23','M61','M97', 'S1','S108','S11','S159','S177','S181','S205', 'S22','S251','S31','S37','S39', 'S4','S5','S50','S51','S74']
l = 10
for i in range(1,l+1):
    classes_rev = classes_rev + [classes[l-i]]
ax.xaxis.set_ticklabels(classes);ax.yaxis.set_ticklabels(classes);
ax.figure.savefig("hybrid_cnn.png")
print('Accuracy: ', score)
print('F1',sklearn.metrics.f1_score(y_true, y_pred, average='weighted'))

train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']
def save_plots(train_acc, val_acc, train_loss, val_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        val_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.show()
    # loss plots
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        val_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()
save_plots(train_acc, val_acc, train_loss, val_loss)
