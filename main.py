import tensorflow as tf
import pandas as pd
import numpy as np
#import torch
import os
import glob
#import PIL
import imageio
import seaborn as sn
import matplotlib.pyplot as plt
#import torchvision.transforms as transforms



from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.layers import concatenate
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.applications import VGG16, ResNet50

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from IPython.display import HTML, display
#%%

def load_signals_images(input_path):
  print(os.listdir(input_path))
  # Директория - это список абсолютных путей ко всем файлам, лежащим в папке
  # Сортируем файлы по длине их названия
  print('[INFO] Сортировка файлов в директории...\n')
  path = input_path+"/*.png"
  directory = sorted(glob.glob(path), key=len) 

  # Далее сортируем их численно (numerically), то есть сортируем численные названия
  # по возрастанию
  directory = sorted(glob.glob(path), 
                    key=lambda x: int(os.path.basename(x).split('.')[0]))

  def get_key(fp):
      filename = os.path.splitext(os.path.basename(fp))[0]
      int_part = filename.split()[0]
      return int(int_part)
  directory = sorted(glob.glob(path), 
                    key=get_key)
  # В конце получаем отсортированную по возрастанию директорию, с которой считываем
  # картинки в массив

  print('[INFO] Загрузка изображений в массив...\n')
  Im = []
  #bar = display(progress(0, len(directory)), display_id=True) # Progress Bar
  counter = 0
  for image_path in directory:
    Im.append(imageio.imread(image_path))
    counter += 1
    #bar.update(progress(counter, len(directory))) # Update Progress
  # Масштабируем  
  Im = np.array(Im)/255.0
  print("[INFO] Датасет загружен")
  return Im

#%%
def load_signals_attributes(input_path, display=False):
  print('[INFO] Загрузка…\n')
  cols = ["N1", "N2", "class"]
  df = pd.read_csv(input_path, sep=" ", header=None, names=cols)
  print('[INFO] Attributes loaded successfully!\n')
  if display == True:
    df.head()
  return df

#%%
  
def results(history, model, testX, testY):
  plt.rcParams['figure.dpi'] = 100
  #dataset_labels=np.array(['Comb', 'Inner', 'Normal', 'Outer', 'Ball'])
  dataset_labels=np.array(['Normal','Inner', 'Outer', 'Ball', 'Comb'])
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(len(history.history['loss']))

  plt.figure(figsize=(8, 8))
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
  

  test_loss, test_acc = model.evaluate(testX, testY, verbose=2)

  # Confusion Matrix
  predictions = model.predict(testX)
  confusion = confusion_matrix(np.argmax(testY, axis=1), 
                              np.argmax(predictions, axis=1))
  print('CONFUSION MATRIX\n', confusion, '\n\n')
  # normalize confustion matrix [0 1]
  # confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
  df_cm = pd.DataFrame(confusion, 
                       dataset_labels, 
                       dataset_labels)
  sn.set(font_scale=1.2) # for label size
  plt.figure(figsize=(4,4))
  sn.heatmap(df_cm, 
             annot=True, 
             annot_kws={"size": 10}, # font size
             fmt = "d",
             #fmt='.2f', # precision (2 digits)
             linewidths=.5,
             cmap="YlGnBu") 
  plt.title(str(model.name))
  plt.ylabel("True Label")
  plt.xlabel("Predicted Label")
  plt.show()

  print('CLASSIFICATION REPORT\n',
        classification_report(np.argmax(testY, axis=1), 
                              np.argmax(predictions, axis=1), 
                              target_names=dataset_labels))
#%%
#data_root='c:\Vika\LongTotal'
data_root='D:\\ibryaeva\\TransferLearning\\LongTotal'


attr = load_signals_attributes(data_root+"/LongTotal.txt")
images = load_signals_images(data_root)

#%% Делим на обучающую и тестовую выборки
split = train_test_split(attr, images, test_size=0.15, random_state=42)
(AttrX, testAttrX, ImagesX, testImagesX) = split

split = train_test_split(AttrX, ImagesX, test_size=0.15, random_state=42)
(trainAttrX,validAttrX, trainImagesX, validImagesX) = split


continuous = ["N1", "N2"]

# Нормализация в диапазоне [0 1]
cs = MinMaxScaler()
trainAttrXnorm = cs.fit_transform(trainAttrX[continuous])
validAttrXnorm = cs.fit_transform(validAttrX[continuous])
testAttrXnorm = cs.transform(testAttrX[continuous])

#zipBinarizer = LabelBinarizer().fit(attr["class"])
#trainY = zipBinarizer.transform(trainAttrX["class"])
#validY = zipBinarizer.transform(validAttrX["class"])
#testY = zipBinarizer.transform(testAttrX["class"])

trainY = trainAttrX["class"]
validY = validAttrX["class"]
testY = testAttrX["class"]

print(trainAttrX.shape)
print(validAttrX.shape)
print(testAttrX.shape)
print(trainImagesX.shape)
print(validImagesX.shape)
print(testImagesX.shape)

#%% 1. Выделение признаков VGG

vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

model = Sequential()

for layer in vgg16_model.layers[:]: # Сколько слоев удаляем с конца
    model.add(layer)    

#model.trainable = False
model.summary()

#%% Выделение признаков из изображений с помощью VGG16

import time
start_time = time.time()

featurestrain = model.predict(trainImagesX)
featuresval = model.predict(validImagesX)
featurestest = model.predict(testImagesX)

featurestrain = featurestrain.reshape((featurestrain.shape[0], 512))
featuresval = featuresval.reshape((featuresval.shape[0],  512))
featurestest = featurestest.reshape((featurestest.shape[0],  512))


print("--- %s seconds ---" % (time.time() - start_time))



#%% добавим числа N1,N2 к вектору признаков
start_time = time.time()
ftrain = np.hstack((featurestrain,trainAttrXnorm))  
fvalid = np.hstack((featuresval,validAttrXnorm))
ftest = np.hstack((featurestest,testAttrXnorm))

# нормируем признаки для дальнейшего использования РСА
from sklearn.decomposition import PCA

X=ftrain
scaler=MinMaxScaler().fit(X)
#normalized_X=(X-X.min())/(X.max()-X.min()) # нормируем их
normalized_X=scaler.fit_transform(X)
normalized_Xval=scaler.transform(fvalid)
normalized_Xtest=scaler.transform(ftest)

print("--- %s seconds ---" % (time.time() - start_time))

pca = PCA().fit(normalized_X)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


pca = PCA(0.99).fit(normalized_X)
pca.n_components_

#%% Понижаем размерность до 231

start_time = time.time()
pca = PCA(n_components=231)

# fit PCA model 
pca.fit(normalized_X)

# transform data onto the first two principal components
X_pca = pca.transform(normalized_X)
X_val=pca.transform(normalized_Xval)
X_test=pca.transform(normalized_Xtest)

print("--- %s seconds ---" % (time.time() - start_time))

#%% t-SNE

from sklearn.manifold import TSNE

data = X_pca
labels = trainY

model = TSNE(n_components=2, random_state=0, n_iter=3000)
# configuring the parameteres
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000

tsne_data = model.fit_transform(data)
tsne_data = np.vstack((tsne_data.T, labels)).T
#%%
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
sn.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()

#%%
from sklearn.preprocessing import LabelBinarizer
Binarizer = LabelBinarizer().fit(trainY)
Y_train = Binarizer.transform(trainY)
Y_val = Binarizer.transform(validY)
Y_test = Binarizer.transform(testY)

#%% Обучение нейронной сети

mlp = Sequential()
mlp.add(Dense(256, input_shape=(231,), activation="relu"))
mlp.add(Dense(32, activation="relu"))
mlp.add(Dense(16, activation="relu"))
mlp.add(Dense(5))

#%%

start_time = time.time()
checkpoint_path='bestmodel.h5'

keras_callbacks   = [
      EarlyStopping(monitor='val_loss', patience=50, verbose=1),
      ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True,  verbose=1)
]

mlp.compile(optimizer='adam', 
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# train the model
print("[INFO] Training model...")
history_mlp = mlp.fit(X_pca, Y_train,
              validation_data=(X_val, Y_val), batch_size=8,
              epochs=500,
              callbacks=[keras_callbacks])

print("--- %s seconds ---" % (time.time() - start_time))

#%%

def results(history, model, testX, testY):
  plt.rcParams['figure.dpi'] = 100
  #dataset_labels=np.array(['Comb', 'Inner', 'Normal', 'Outer', 'Ball'])
  dataset_labels=np.array(['Normal','Inner', 'Outer', 'Ball', 'Comb'])
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(len(history.history['loss']))

  plt.figure(figsize=(14, 8))
  
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
  
 
 
  

  test_loss, test_acc = model.evaluate(testX, testY, verbose=2)

  # Confusion Matrix
  predictions = model.predict(testX)
  confusion = confusion_matrix(np.argmax(testY, axis=1), 
                              np.argmax(predictions, axis=1))
  print('CONFUSION MATRIX\n', confusion, '\n\n')
  # normalize confustion matrix [0 1]
  # confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
  df_cm = pd.DataFrame(confusion, 
                       dataset_labels, 
                       dataset_labels)
  sn.set(font_scale=1.2) # for label size
  plt.figure(figsize=(4,4))
  sn.heatmap(df_cm, 
             annot=True, 
             annot_kws={"size": 10}, # font size
             fmt = "d",
             #fmt='.3f', # precision (3 digits)
             linewidths=.5,
             cmap="YlGnBu") 
  #plt.title(str(model.name))
  plt.ylabel("True Label")
  plt.xlabel("Predicted Label")
  plt.show()

  print('CLASSIFICATION REPORT\n',
        classification_report(np.argmax(testY, axis=1), 
                              np.argmax(predictions, axis=1), 
                              target_names=dataset_labels,  digits=3))

#%%

best_model_mlp = load_model(checkpoint_path)

results(history = history_mlp, 
        model = best_model_mlp, 
        testX = X_test, 
        testY = Y_test)

#%% 2. Выделение признаков ResNet
RN_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

for layer in RN_model.layers:
    layer.trainable = False

model = RN_model

model.summary()

#%% Выделение признаков из изображений с помощью ResNet

import time
start_time = time.time()

featurestrain = model.predict(trainImagesX)
featuresval = model.predict(validImagesX)
featurestest = model.predict(testImagesX)

featurestrain = featurestrain.reshape((featurestrain.shape[0], 2048))
featuresval = featuresval.reshape((featuresval.shape[0],  2048))
featurestest = featurestest.reshape((featurestest.shape[0],  2048))

print("--- %s seconds ---" % (time.time() - start_time))



#%% добавим числа N1,N2 к вектору признаков
start_time = time.time()
ftrain = np.hstack((featurestrain,trainAttrXnorm))  
fvalid = np.hstack((featuresval,validAttrXnorm))
ftest = np.hstack((featurestest,testAttrXnorm))

# нормируем признаки для дальнейшего использования РСА
from sklearn.decomposition import PCA

X=ftrain
scaler=MinMaxScaler().fit(X)
#normalized_X=(X-X.min())/(X.max()-X.min()) # нормируем их
normalized_X=scaler.fit_transform(X)
normalized_Xval=scaler.transform(fvalid)
normalized_Xtest=scaler.transform(ftest)

print("--- %s seconds ---" % (time.time() - start_time))

pca = PCA().fit(normalized_X)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


pca = PCA(0.99).fit(normalized_X)
pca.n_components_

#%% Понижаем размерность до 48

start_time = time.time()
pca = PCA(n_components=48)

# fit PCA model 
pca.fit(normalized_X)

# transform data onto the first two principal components
X_pca = pca.transform(normalized_X)
X_val=pca.transform(normalized_Xval)
X_test=pca.transform(normalized_Xtest)

print("--- %s seconds ---" % (time.time() - start_time))

#%% t-SNE

from sklearn.manifold import TSNE

data = X_pca
labels = trainY

model = TSNE(n_components=2, random_state=0, n_iter=3000, perplexity=50)
# configuring the parameteres
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000

tsne_data = model.fit_transform(data)
tsne_data = np.vstack((tsne_data.T, labels)).T
#%%
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
sn.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()


#%% Обучение нейронной сети

mlp = Sequential()
mlp.add(Dense(256, input_shape=(48,), activation="relu"))
mlp.add(Dense(32, activation="relu"))
mlp.add(Dense(16, activation="relu"))
mlp.add(Dense(5))

#%%
acc=[]
t=[]
for N in range(5,501,10):
 
    acc1=[]
    t1=[]
    
    for i in range(10):
        
    
        start_time = time.time()
        mlp = Sequential()
        mlp.add(Dense(N, input_shape=(48,), activation="relu"))
        #mlp.add(Dense(32, activation="relu"))
        mlp.add(Dense(5))
    
        
        checkpoint_path='bestmodel.h5'
        
        keras_callbacks   = [
              EarlyStopping(monitor='val_accuracy', patience=50, verbose=1),
              ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True,  verbose=1)
        ]
        
        mlp.compile(optimizer='adam', 
                    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        
        # train the model
        print("[INFO] Training model...")
        history = mlp.fit(X_pca, Y_train,
                      validation_data=(X_val, Y_val), batch_size=32,
                      epochs=500,
                      callbacks=[keras_callbacks])
        
        
        acc1.append(np.mean(history.history['val_accuracy']))
        t1.append(time.time() - start_time)
    
    acc.append(np.mean(acc1))
    t.append(np.mean(t1))

#%%


plt.plot(range(5,501,10),acc)
plt.xlim([5,500])
plt.xlabel('N')
plt.ylabel('accuracy')
plt.grid()

plt.plot(range(5,501,10),t)
plt.xlim([5,500])
plt.xlabel('N')
plt.ylabel('time, seconds')
plt.grid()


#%% Обучение нейронной сети N=250

mlp = Sequential()
mlp.add(Dense(250, input_shape=(48,), activation="relu"))
mlp.add(Dense(5))

checkpoint_path='bestmodel.h5'
        
keras_callbacks   = [
              EarlyStopping(monitor='val_accuracy', patience=50, verbose=1),
              ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True,  verbose=1)
        ]
        
mlp.compile(optimizer='adam', 
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        
        # train the model
print("[INFO] Training model...")
history = mlp.fit(X_pca, Y_train,
                      validation_data=(X_val, Y_val), batch_size=32,
                      epochs=500,
                      callbacks=[keras_callbacks])

#%%

best_model_mlp = load_model(checkpoint_path)

results(history = history, 
        model = best_model_mlp, 
        testX = X_test, 
        testY = Y_test)

#%%
acc=[]
t=[]
for N1 in range(5,501,10):
    print(N1)
    
    for N2 in range(5,501,10):
 
        acc1=[]
        t1=[]
        
        for i in range(2):
            
        
            start_time = time.time()
            mlp = Sequential()
            mlp.add(Dense(N1, input_shape=(48,), activation="relu"))
            mlp.add(Dense(N2, activation="relu"))
            mlp.add(Dense(5))
        
            
            checkpoint_path='bestmodel.h5'
            
            keras_callbacks   = [
                  EarlyStopping(monitor='val_accuracy', patience=50, verbose=1),
                  ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True,  verbose=1)
            ]
            
            mlp.compile(optimizer='adam', 
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
            
            # train the model
            print("[INFO] Training model...")
            history = mlp.fit(X_pca, Y_train,
                          validation_data=(X_val, Y_val), batch_size=2048,
                          epochs=500,
                          callbacks=[keras_callbacks])
            
            
            acc1.append(np.mean(history.history['val_accuracy']))
            t1.append(time.time() - start_time)
    
        acc.append(np.mean(acc1))
        t.append(np.mean(t1))

#%%

M=np.array(acc).reshape(50,50)

fig,ax=plt.subplots(1,1)
img=ax.imshow(M, origin='lower', cmap='brg')

ax.set_xticks([0,10,20,30,40])
ax.set_xticklabels([5,100,200,300,400])
ax.set_yticks([0,10,20,30,40])
ax.set_yticklabels([5,100,200,300,400])

ax.set_xlabel('N1')
ax.set_ylabel('N2')
fig.colorbar(img, label='Accuracy', orientation='horizontal', fraction=0.046, pad=0.2)





#%%
def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)

	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs

		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
    

	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(16)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x) 

	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = Dense(8)(x)
	x = Activation("relu")(x)

	# check to see if the regression node should be added
	if regress:
		x = Dense(5)(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model  

#%%

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
print('[INFO] Creating CNN...')
model_cnn = create_cnn(32, 32, 3, regress=True)
print('[INFO] Compiling model...')

checkpoint_path='bestcnn.h5'
keras_callbacks   = [
      EarlyStopping(monitor='val_loss', patience=50, verbose=1),
      ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
]

opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-3 / 200)
model_cnn.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

start_time = time.time()

history_model_cnn = model_cnn.fit(
              x=trainImagesX, y=Y_train,
              validation_data=(validImagesX, Y_val),
              batch_size =2048,
              epochs=2000,
              callbacks=[keras_callbacks])
              
              
print("--- %s seconds ---" % (time.time() - start_time))

              
#%%

best_model_cnn = load_model(checkpoint_path)

# это лучшая модель
results(history_model_cnn, 
        best_model_cnn, 
        testImagesX, 
        Y_test)




