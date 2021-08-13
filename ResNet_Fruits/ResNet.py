"""
Create on Mon 9 14:10:05 2021
@author: MBI
Descipcion:
Estructura de la red ResNet para el modelo
"""
#%% Modulos
from tensorflow.keras.layers import Conv2D,Activation,BatchNormalization,Add,Flatten,AveragePooling2D,Dense
from tensorflow.keras import Input,Model
from tensorflow.keras.regularizers import l2

class ResNetV2():
    def __init__(self,input_dim=None,output_dim=None,depth=0): # Valores a entrar tamaño de entrada,salida y profundidad de la red
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_block = 0

        if (depth - 2) % 9 != 0:
            raise ValueError('La profundidad debe ser 9n+2')
        self.n_block = int((depth - 2)/9)
        self.resnet = self.model()

    def resnetLayer(self,inputs,n_filters=16,kernel=3,stride=1,activation='relu',batch_nor=True,conv_firt=True):
        conv = Conv2D(filters=n_filters,kernel_size=kernel,strides=stride,padding='same',kernel_regularizer=l2(1e-4))
        x = inputs
        if conv_firt:
            x = conv(x)
            if batch_nor:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_nor:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def model(self):
        n_filters = 16
        inputs = Input(shape=self.input_dim)
        x = self.resnetLayer(inputs=inputs)
        for stack in range(3):
            for nblock in range(self.n_block):
                activation = 'relu'
                batch_nor = True
                strides = 1
                if stack == 0:
                    n_filters_out = n_filters * 4
                    if nblock == 0:
                        activation = None
                        batch_nor = False
                else:
                    n_filters_out = n_filters * 2
                    if nblock == 0:
                        strides = 2

                y = self.resnetLayer(x,n_filters=n_filters,kernel=1,stride=strides,activation=activation,batch_nor=batch_nor,conv_firt=False)
                y = self.resnetLayer(y,n_filters=n_filters,conv_firt=False)
                y = self.resnetLayer(y,n_filters=n_filters_out,conv_firt=False,kernel=1)

                if nblock == 0:
                    x = self.resnetLayer(x,n_filters=n_filters_out,kernel=1,activation=None,batch_nor=False,stride=strides)

                x  = Add()([x,y])

            n_filters = n_filters_out

        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(units=self.output_dim,activation='softmax',kernel_initializer='he_normal')(y)
        model = Model(inputs,outputs)
        return model






