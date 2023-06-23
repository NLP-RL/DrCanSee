
from google.colab import drive
drive.mount('/content/drive')

import pickle

with open('drive/MyDrive/final_ee/goal_set.p','rb') as f:
    train_data = pickle.load(f)

s_list = ['Swollen eye',  'Mouth ulcer',  'Skin dryness',  'Neck swelling',  'Skin irritation', 'Swollen tonsils', 'Knee swelling', 'Redness in ear', 'Dry scalp', 'Skin rash', 'Itchy eyelid', 'Hand lump', 'Skin growth', 'Eye redness', 'Foot swelling', 'Lip swelling',  'Eyelid rash']
# print(s_list.index('Neck swelling'))
l = []
temp = []
sno = []
sno_dict = {}
for symptom in s_list:
  for i in range(24000):
    traind = train_data['train'][i]['goal']
    # print(traind)
    try:
      traind.pop("request_slots")
    except:
      pass
    for key in traind:
      x = traind[key]
      # print(x)
      # Swollen eye
      for key2 in x:
        # print(key2)
        if key2 == symptom:
          sno.append(i)
          break
  if len(sno)>0:
    sno_dict[symptom] = sno[:100]
  sno = []

for i in sno_dict:
  # print(sno_dict[i])
  print(i, len(sno_dict[i]))

fs = {}
cnt = 0
for symptom in sno_dict:
  # print(symptom)
  for i in sno_dict[symptom]:
    td = train_data['train'][i]['goal']
    # print(td)
    for key in traind:
      x = td[key]
      # print(x)
      # Swollen eye
      for key2 in x:
        # print(key2)
        if key2==symptom:
          break
        temp.append(key2)

    l.append([temp, symptom, cnt])
    temp = []
  cnt = cnt + 1
  fs[symptom] = l
  l = []
# print(sno_dict['Swollen eye'])
print(fs['Mouth ulcer'])
print(len(fs['Mouth ulcer']))
# print(len(fs))

print(fs['Swollen eye'])

for i in fs:
  # print(fs[i])
  for j in fs[i]:
    if j[0] == []:
      fs[i].remove(j)
      print('yes')

for i in fs:
  # print(sno_dict[i])
  print(i, len(fs[i]))

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import matplotlib
matplotlib.style.use('ggplot')

IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = 'drive/MyDrive/context_aware/train'
VALID_DATA_DIR = 'drive/MyDrive/context_aware/test'

import copy
final_data  = copy.deepcopy(fs)

import os
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)
path = "drive/MyDrive/context_aware/train/S51"
dirlist = sorted_alphanumeric(os.listdir(path))
print(len(dirlist))
fdirlist = dirlist + dirlist
print(len(fdirlist))
print(fdirlist)

symptom = 'Lip swelling'
for i in range(len(final_data[symptom])):
  print(final_data[symptom][i])
  temp_list = final_data[symptom][i]
  temp_list[1] = 'S51/'+fdirlist[i]
  final_data[symptom][i] = temp_list
  print(final_data[symptom][i])
  print('\n\n')
# final_data[symptom]

for i in final_data:
  print(final_data[i])
pickle.dump(final_data, open('train_ca_v1.p', 'wb'))

"""# Test_data"""

import pickle

with open('drive/MyDrive/final_ee/goal_set.p','rb') as f:
    train_data = pickle.load(f)

s_list = ['Swollen eye',  'Mouth ulcer',  'Skin dryness, peeling, scaliness, or roughness',  'Neck swelling',  'Skin irritation', 'Swollen or red tonsils', 'Knee swelling', 'Redness in ear', 'Dry or flaky scalp', 'Skin rash', 'Itchy eyelid', 'Hand or finger lump or mass', 'Skin growth', 'Eye redness', 'Foot or toe swelling', 'Lip swelling',  'Eyelid lesion or rash']
# print(s_list.index('Neck swelling'))
l = []
temp = []
sno = []
sno_dict = {}
for symptom in s_list:
  for i in range(6000):
    traind = train_data['test'][i]['goal']
    # print(traind)
    try:
      traind.pop("request_slots")
    except:
      pass
    for key in traind:
      x = traind[key]
      # print(x)
      # Swollen eye
      for key2 in x:
        # print(key2)
        if key2 == symptom:
          sno.append(i)
          break
  if len(sno)>0:
    sno_dict[symptom] = sno[:10]
  sno = []

for i in sno_dict:
  # print(sno_dict[i])
  print(i, len(sno_dict[i]))

fs = {}
cnt = 0
for symptom in sno_dict:
  # print(symptom)
  for i in sno_dict[symptom]:
    td = train_data['test'][i]['goal']
    # print(td)
    for key in traind:
      x = td[key]
      # print(x)
      # Swollen eye
      for key2 in x:
        # print(key2)
        if key2==symptom:
          break
        temp.append(key2)

    l.append([temp, symptom, cnt])
    temp = []
  cnt = cnt + 1
  fs[symptom] = l
  l = []
# print(sno_dict['Swollen eye'])
print(fs['Mouth ulcer'])
print(len(fs['Mouth ulcer']))
# print(len(fs))

print(fs['Swollen eye'])

for i in fs:
  # print(fs[i])
  for j in fs[i]:
    if j[0] == []:
      fs[i].remove(j)
      print('yes')
big_list = []
fdict = {}
for i in fs:
  # print(sno_dict[i])
  fdict[i] = len(fs[i])
  print(i, len(fs[i]))
  big_list.append(i)
print(fdict)
pickle.dump(fdict, open('SD_test_distribution.p', 'wb'))

import matplotlib.pyplot as plt
import os
import tensorflow as tf
import matplotlib
matplotlib.style.use('ggplot')

import copy
final_data  = copy.deepcopy(fs)

import os
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

bd = {'Swollen eye': 'S1',  'Mouth ulcer': 'S108',  'Skin dryness, peeling, scaliness, or roughness':'S11',  'Neck swelling': 'S159',  'Skin irritation': 'S177', 'Swollen or red tonsils': 'S181', 'Knee swelling' : 'S205', 'Redness in ear' : 'S22', 'Dry or flaky scalp': 'S251', 'Skin rash': 'S31', 'Itchy eyelid' : 'S37', 'Hand or finger lump or mass':'S39', 'Skin growth':'S4', 'Eye redness':'S5', 'Foot or toe swelling':'S50', 'Lip swelling': 'S51',  'Eyelid lesion or rash':'S74'}
for index in final_data:
  path = "drive/MyDrive/context_aware/test/" + bd[index]
  dirlist = sorted_alphanumeric(os.listdir(path))
  print(len(dirlist))
  fdirlist = dirlist + dirlist
  print(len(fdirlist))
  print(fdirlist)
  symptom = index
  for i in range(len(final_data[symptom])):
    print(final_data[symptom][i])
    temp_list = final_data[symptom][i]
    temp_list[1] = bd[index]+'/'+fdirlist[i]
    final_data[symptom][i] = temp_list
    print(final_data[symptom][i])
    print('\n\n')

for i in final_data:
  print(final_data[i])
pickle.dump(final_data, open('test_ca_v1.p', 'wb'))



"""# Model"""

!pip install transformers

import io
import re
from transformers import AutoTokenizer
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from transformers import pipeline

#Bert
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('feature-extraction',model=model_name, tokenizer=tokenizer)

def lambda_func(row):
    tokens = tokenizer(row[0],padding=True,truncation=True,return_tensors="pt")
    if len(tokens['input_ids'])>512:
        tokens = re.split(r'\b', row[0])
        tokens= [t for t in tokens if len(t) > 0 ]
        row[0] = ''.join(tokens[:512])
    row['vectors'] = classifier(row[0])[0][0]
    return row

def process(progress_notes):
    progress_notes = progress_notes.apply(lambda_func, axis=1)
    return progress_notes

from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import numpy as np
a = ['cold[SEP]rash']
progress_notes_a=pd.DataFrame(a)
# print(progress_notes_a)
progress_notes_a = process(progress_notes_a)
progress_notes_a = progress_notes_a['vectors'][0]
progress_notes_a = np.array(progress_notes_a)

# progress_notes_a = np.reshape(progress_notes_a, (1,768))
# print(progress_notes_a)
b = ['rash']
progress_notes_b=pd.DataFrame(b)
progress_notes_b = process(progress_notes_b)
progress_notes_b = progress_notes_b['vectors'][0]
# progress_notes_b = np.array(progress_notes_b)

# progress_notes_b = np.reshape(progress_notes_b, (1,768))
# cosine_similarity(progress_notes_a, progress_notes_b)
from scipy import spatial

result = 1 - spatial.distance.cosine(progress_notes_a, progress_notes_b)
result

def embedding(emb):
  pn=pd.DataFrame([emb])
  pn = process(pn)
  pn = pn['vectors'][0]
  pn = np.array(pn)
  return pn

array = embedding('cold')
print(len(array))

l = ['1','2']
print(l[-6:])
print('[sep]'.join(l))
dic = {'3':'h'}
for i in l:
  if i in dic:
    print('yes')

import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import os
import pandas as pd
from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import albumentations as A
from PIL import Image

import pickle
with open('drive/MyDrive/context_aware/train_ca_v1.p','rb') as f:
    train_data = pickle.load(f)
with open('drive/MyDrive/context_aware/test_ca_v1.p','rb') as f:
    test_data = pickle.load(f)
print(len(train_data))
# print(train_data)
print(len(test_data))
train_list = []
test_list = []
for i in train_data:
  for j in train_data[i]:
    train_list.append(j)
for i in test_data:
  for j in test_data[i]:
    test_list.append(j)
# print(train_list[23])
print(test_list[12])
# for i in range(len(test_list)):
#   print(test_list[i])
# s_list = ['Swollen eye',  'Mouth ulcer',  'Skin dryness',  'Neck swelling',  'Skin irritation', 'Swollen tonsils', 'Knee swelling', 'Redness in ear', 'Dry scalp', 'Skin rash', 'Itchy eyelid', 'Hand lump', 'Skin growth', 'Eye redness', 'Foot swelling', 'Lip swelling',  'Eyelid rash']

class Symptoms(Dataset):
    def __init__(self, data_list, root_dir,context_no, transform=None):
        self.annotations = data_list
        self.context_no = context_no
        # self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def embedding(emb):
        pn=pd.DataFrame([emb])
        pn = process(pn)
        pn = pn['vectors'][0]
        pn = np.array(pn)
        return pn

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations[index][1])
        image = np.array(Image.open(img_path).convert('RGB'))
        y_label = torch.tensor(int(self.annotations[index][2]))
        emb_list = self.annotations[index][0]
        x = self.context_no
        temp_list = emb_list[-x:]
        change_dic = {'Skin dryness, peeling, scaliness, or roughness': 'Skin dryness',
                        'Swollen or red tonsils':'Swollen tonsils',  'Dry or flaky scalp': 'Dry scalp',
                        'Hand or finger lump or mass': 'Hand lump',
                       'Foot or toe swelling': 'Foot swelling',  'Eyelid lesion or rash': 'Eyelid rash'}
        for i in range(len(temp_list)):
          if temp_list[i] in change_dic:
            temp_list[i] = change_dic[temp_list[i]]
        seperator = '[SEP]'
        final_embedding_string = seperator.join(temp_list)
        # print(final_embedding_string)
        emb_array = embedding(final_embedding_string)
        emb_array = torch.tensor(emb_array)

        transform = A.Compose(
        [A.Resize(width=224, height=224)],
        )
        augmentations = transform(image=image)
        image = augmentations["image"]


        if self.transform:
            image = self.transform(image)

        return (image, y_label, emb_array)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 3
num_classes = 17

batch_size = 32


# Load Data
train_dataset = Symptoms(
    data_list = train_list,
    root_dir='drive/MyDrive/context_aware/train',
    context_no = 3,
    transform=transforms.ToTensor(),
)
test_dataset = Symptoms(
    data_list = test_list,
    root_dir='drive/MyDrive/context_aware/test',
    context_no = 3,
    transform=transforms.ToTensor(),
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



import torchvision.models as models
densenet = models.densenet161(pretrained=True,).to(device)
for param in densenet.parameters():
    param.requires_grad = False
# print(densenet)

import torch.nn.functional as F
class Dense(nn.Module):
    def __init__(self, original_model):
        super(Dense, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.layer = nn.Linear(768, 32)
        self.conv1 = nn.Conv2d(2208, 1024,  3, 1,  bias=False, padding="valid")
        self.conv2 = nn.Conv2d(1024, 1024,  2, 1, bias=False, padding="valid")
        # self.pool = nn.MaxPool2d(4, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.linear = nn.Linear(1792, 512)
        self.output = nn.Linear(512,17)


    def forward(self, x, e):
        # print(x.shape)
        x = self.features(x)
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        # print(x.shape)
        a,b,c,d = x.shape
        x = torch.reshape(x, (a, -1))
        # print(x.shape)
        x = torch.cat([x, e], dim=1)
        # print(x.shape)
        x = F.relu(self.linear(x))
        # print(x.shape)
        x = self.output(x)
        # print(x.shape)
        return x

# x = torch.randn((32, 3, 224, 224)).to(device)
# e = torch.randn((32, 768)).to(device)
den_conv = Dense(densenet).to(device)
# outputs = den_conv(x, e)
# print(outputs.shape)
# preds = densenet(x)
# print(preds.shape)

pytorch_total_params = sum(p.numel() for p in den_conv.parameters())
trainable = sum(p.numel() for p in den_conv.parameters() if p.requires_grad)
print('trainable',trainable)
print('total',pytorch_total_params)

from tqdm import tqdm
# Loss and optimizer
learning_rate = 1e-3
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(den_conv.parameters(), lr=learning_rate, weight_decay = 1e-5)

# decayRate = 0.96
# my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

num_epochs = 15

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y, e in loader:
            x = x.to(device=device)
            e = e.to(device=device)
            y = y.to(device=device)

            scores = model(x, e.float())
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)


    model.train()
    return num_correct/num_samples

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


# Train Network
for epoch in range(num_epochs):
    print("\n\nCurrent EPOCH:", epoch)
    for batch_idx, (data, targets, embeddings) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        embeddings = embeddings.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = den_conv(data, embeddings.float())
        loss = criterion(scores, targets)
        if batch_idx==46:
          print(f'Loss at step {batch_idx} = ',loss)
        if batch_idx%5 ==0:
          print(f'Loss at step {batch_idx} = ',loss)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    # if epoch%3==0:
    #   print(f"Accuracy at EPOCH {epoch} on training set: {check_accuracy(train_loader, den_conv)*100:.2f}")
    print(f"Accuracy at EPOCH {epoch} on test set: {check_accuracy(test_loader, den_conv)*100:.2f}")
    # if epoch%1==0:
    #   fname  = 'drive/MyDrive/context_aware/ca_weights/v12/'+'v12_checkpoint_'+str(epoch+10)+'.pth.tar'
    #   save_checkpoint(den_conv, optimizer, fname)

# Check accuracy on training & test to see how good our model

learning_rate = 1e-3
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(den_conv.parameters(), lr=learning_rate, weight_decay = 1e-5)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
path = 'drive/MyDrive/context_aware/ca_weights/v12/v12_checkpoint_13.pth.tar'
load_checkpoint(path, den_conv,optimizer, learning_rate)

my_list = [[['Vomiting', 'Skin irritation', 'Skin rash'],'S108/S108_3.jpg', 1], [['Eye redness', 'Knee swelling', 'Back pain'],'S108/S108_3.jpg', 1]]

my_dataset = Symptoms(
    data_list = my_list,
    root_dir='drive/MyDrive/context_aware/test',
    context_no = 3,
    transform=transforms.ToTensor(),
)

my_loader = DataLoader(dataset=my_dataset, batch_size=batch_size, shuffle=False)

import numpy
y_pred = []
y_true = []
def check_accuracy(loader, model,y_true,y_pred):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y, e in loader:
            x = x.to(device=device)
            e = e.to(device=device)
            y = y.to(device=device)
            # print(y.shape)
            lis1 = numpy.array(y.cpu())
            lis1 = lis1.tolist()
            # print(lis1)
            y_true = y_true + lis1
            # print(y_true)
            scores = model(x, e.float())
            _, predictions = scores.max(1)
            lis2 = numpy.array(predictions.cpu())
            lis2 = lis2.tolist()
            y_pred = y_pred + lis2


            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)


    model.train()
    return num_correct/num_samples, y_true, y_pred

a, y_true, y_pred =check_accuracy(test_loader, den_conv,y_true, y_pred)
print(f"Accuracy on test set: {a*100:.2f}")
print(y_true)
print(len(y_true))
print(y_pred)
print(len(y_pred))
for i in range(len(y_true)):
  print(y_true[i],'  ', y_pred[i],' ', i)





# k = [ 'S1',   'S108',  'S11',   'S159',   'S177', 'S181', 'S205',  'S22', 'S251', 'S31',  'S37', 'S39', 'S4', 'S5', 'S50',  'S51',  'S74']
k = ['Swollen eye',  'Mouth ulcer',  'Skin dryness',  'Neck swelling',  'Skin irritation', 'Swollen tonsils', 'Knee swelling', 'Redness in ear', 'Dry scalp', 'Skin rash', 'Itchy eyelid', 'Hand lump', 'Skin growth', 'Eye redness', 'Foot swelling', 'Lip swelling',  'Eyelid rash']

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns



print('true classes:',y_true)

score = accuracy_score(y_true, y_pred)
print('Classification Report: ',classification_report(y_true,y_pred))
cm = confusion_matrix(y_true, y_pred)
f,ax= plt.subplots(figsize=(15, 15))
sns.heatmap(cm, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels',fontsize = 28);ax.set_ylabel('True labels',fontsize = 28);
ax.set_title('Confusion Matrix');
classes_rev = []

classes = k
l = len(k)
for i in range(1,l+1):
    classes_rev = classes_rev + [classes[l-i]]
ax.xaxis.set_ticklabels(classes,fontsize = 5, );ax.yaxis.set_ticklabels(classes, fontsize = 5);
ax.figure.savefig("hybrid_cnn.png")
print('Accuracy: ', score)
print('F1',sklearn.metrics.f1_score(y_true, y_pred, average='weighted'))

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
print(len(multimodal_symp))
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
for i in new_sd:
  print(i)
pickle.dump(new_sd, open('SD_DesneNet169CNN_contextaware_v12.p', 'wb'))
