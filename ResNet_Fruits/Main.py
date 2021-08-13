"""
Create on Mon 9 14:10:05 2021
@author: MBI
Descripcion: Estructura base que se alimenta de la red ResNet
"""
#%% Modulos
import tensorflow as tf
import matplotlib.pyplot as plt
import os,pathlib
import numpy as np
from tensorflow.keras import  callbacks
from tensorflow.keras.utils import  plot_model
from tensorflow.keras.optimizers import  RMSprop
from ResNet import ResNetV2
import cv2 as cv
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(42)

#%% Arquitectura del modelo

class ModelArq():
    def __init__(self,input_shape,output_shape,depth=38):
        self.path_model = os.getcwd()
        self.history = None
        self.checkpoint = callbacks.ModelCheckpoint(filepath=self.path_model+'\\'+'ResNetV2.h5',monitor='val_acc',verbose=1)
        self.scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss',min_lr=0.5e-6,factor=np.sqrt(0.1),patience=2,mode='min')
        self.stop_epoch = callbacks.EarlyStopping(monitor='val_acc',patience=2,min_delta=0.05,mode='max')

        self.model = ResNetV2(input_shape,output_shape,depth)
        self.model.resnet.compile(
            optimizer = RMSprop(learning_rate=0.001,rho=0.90),
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])
        self.model.resnet.summary()
        plot_model(model=self.model.resnet, show_shapes=True, to_file=self.path_model+'\\'+'ResNetV2.png')



    def train_model(self,ds_train,ds_val,num_epochs):
        self.history = self.model.resnet.fit(
            ds_train,
            validation_data=ds_val,
            epochs=num_epochs,
            callbacks=[self.checkpoint,self.scheduler,self.stop_epoch]
        )

    
    def show_performer(self):
        figure = plt.figure(figsize=(12,8))
        plt.suptitle('Metrics (Training and Validation',fontsize=15)
        axis_1 = figure.add_subplot(121)
        axis_2 = figure.add_subplot(122)

        axis_1.set_title('Training and Validation (Losses)',fontsize=10)
        axis_1.plot(self.history.history['loss'],color='red',label='Training losses')
        axis_1.plot(self.history.history['val_loss'],color='blue',label='Validation losses')
        axis_1.set_ylim((0,1))
        axis_1.set_xlabel('Epochs')
        axis_1.set_ylabel('Values')
        axis_1.legend(loc='upper right')
        
        axis_2.set_title('Training and Validation (accuracy)',fontsize=10)
        axis_2.plot(self.history.history['accuracy'],color='red',label='Training accuracy')
        axis_2.plot(self.history.history['val_accuracy'],color='blue',label='Validation accuracy')
        axis_2.set_ylim((0, 1))
        axis_2.set_ylabel('Values')
        axis_2.set_xlabel('Epochs')
        axis_2.legend(loc='upper right')
       
        plt.show()
        try:
            figure.savefig(self.path_model+'\\'+'Metrics_ResNetV2.jpg',dpi=100,format='jpg')
        except :
            raise ValueError('Not saved image')


# %% Procesamiento de datos

path_train = pathlib.Path( "C:\\Users\\MBI\\Documents\\Python_Scripts\\Datasets\\fruits-360\\Training")
path_val = pathlib.Path( "C:\\Users\\MBI\\Documents\\Python_Scripts\\Datasets\\fruits-360\\Validation")

imagen_train = len(list(path_train.glob('*/*.jpg')))
imagen_val = len(list(path_val.glob('*/*.jpg')))

dic_class = {}
list_label_train = []
list_train_data = []
list_label_val = []
list_val_data = []
num_epochs = 10
batch_size = 60
shape = (100,100,3)

for id, cls in enumerate(os.listdir(path_train),0):
    dic_class[id] = cls

class_count = len(dic_class.keys())

def preprocesImage(path,labels,shape=(100,100),mode='train'):
    file_img = tf.io.read_file(path)
    image = tf.image.decode_jpeg(file_img,channels=3)
    image = tf.image.resize(image,shape)
    image /= 255.0
    image = image * 2 - 1.0


    if mode == 'trian':
        image_crop = tf.image.random_crop(value=image,size=(40,40,3))
        image_rizased = tf.image.resize(image_crop,size=shape)
        image_b = tf.image.random_brightness(image_rizased,max_delta=32. / 255.)
        image_cont = tf.image.random_contrast(image_b,lower=0.5,upper=1.5)
        image_gama = tf.image.random_saturation(image_cont,lower=0.5,upper=1.5)
        image_flip = tf.image.random_flip_left_right(image_gama)
        return image_flip, labels
    else:
        image_crop = tf.image.crop_to_bounding_box(image,offset_height=4,offset_width=4,target_height=90,target_width=90)
        image_rizased = tf.image.resize(image_crop,size=shape)
        return image_rizased, labels


def load_data(path,list_data,list_label):
    # Guarda las rutas de las imagenes en una lista asociada a una etiqueta numerica.
    for id, cls in enumerate(os.listdir(path),0):
        dirction = os.path.join(path,cls)
        for image in os.listdir(path=dirction):
            list_data.append(dirction + '\\' + image)
            list_label.append(id)


load_data(path_train,list_train_data,list_label_train)
load_data(path_val,list_val_data,list_label_val)

ds_train = tf.data.Dataset.from_tensor_slices((list_train_data,list_label_train))
ds_val = tf.data.Dataset.from_tensor_slices((list_val_data,list_label_val))

ds_train = ds_train.shuffle(buffer_size=imagen_train,reshuffle_each_iteration=True)
ds_val = ds_val.shuffle(buffer_size=imagen_val,reshuffle_each_iteration=True)

dataset_train = ds_train.map(lambda path,labels: preprocesImage(path,labels))
dataset_val = ds_val.map(lambda path,labels: preprocesImage(path,labels,mode='val'))
#dataset_test = ds_test.map(lambda path,labels: preprocesImage(path,labels,mode='test' ))

ATOTUNE = tf.data.experimental.AUTOTUNE
dataset_train = dataset_train.cache().prefetch(buffer_size=ATOTUNE).shuffle(buffer_size=imagen_train,reshuffle_each_iteration=True).batch(batch_size=batch_size)
dataset_val = dataset_val.cache().prefetch(buffer_size=ATOTUNE).shuffle(buffer_size=imagen_val).batch(batch_size=batch_size)
#dataset_test = dataset_test.cache().prefetch(buffer_size=ATOTUNE).batch(batch_size=batch_size)

#%% Instace Class

resnet = ModelArq(shape,class_count,20)

#%% Train Model
resnet.train_model(dataset_train,dataset_val,num_epochs)

#%% Plot Model
resnet.show_performer()

#%% Evaluation Model
evaluation = resnet.model.resnet.evaluate(dataset_val)
tf.print(evaluation)

#%% Test Model

modelo = tf.keras.models.load_model('ResNetV2.h5')

test_path = pathlib.Path("C:\\Users\\MBI\\Documents\\Python_Scripts\\Datasets\\fruits-360\\Test")

def load_image(image_path):
    image = cv.imread(image_path)
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    return image

def test_model(path_list,name_class,modelo):
    figure = plt.figure(figsize=(12,8))

    for path in os.listdir(path=path_list):
        path_img = os.path.join(path_list,path)    
        image = tf.io.read_file(path_img)
        image = tf.image.decode_jpeg(image,channels=3)
        image = tf.image.resize(image,(100,100))
        image = tf.expand_dims(image,axis=0)
        image /= float(255.0)
        image = image * int(2) - float(1.0)
    
        pred = modelo.predict(image)
        conf = np.max(pred[0])
        name = tf.argmax(pred,axis=1).numpy()[0]

        plt.title('{} -- {}'.format(name_class[name],round(conf * 100,2)))
        plt.imshow(load_image(path_img))
        plt.axis('off')
        plt.show()

test_model(test_path,dic_class,modelo)
















