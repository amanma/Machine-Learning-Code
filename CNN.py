CNN

!pip install tensorflow
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#Preprocessing the training dataset
train_datagen = ImageDataGenerator(rescale = 1./255, #Data Normalization
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True) 
training_set = train_datagen.flow_from_directory (r"D:\MBA - Balaji\Sem 3\Deep Learning\CNN Dataset\dataset\training_set",
                                                 target_size=(64,64),
                                                 batch_size = 32,
                                                 class_mode='binary')

#Preprocessing the test dataset
test_datagen = ImageDataGenerator(rescale = 1./255, #Data Normalization
                                  ) 
test_set = train_datagen.flow_from_directory (r"D:\MBA - Balaji\Sem 3\Deep Learning\CNN Dataset\dataset\test_set",
                                                 target_size=(64,64),
                                                 batch_size = 32,
                                                 class_mode='binary')


#Initializing CNN
cnn = tf.keras.models.Sequential()

#Add convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,3]))

#Add pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides = 2))

#Add another convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

#Add pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides = 2))

#Add flattening layer
cnn.add(tf.keras.layers.Flatten())

#Add fully connected layer
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#Add output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


#Compiler
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Training the CNN on the training set and evaluating it on the test set

cnn.fit(x=training_set, validation_data=test_set, epochs = 25)

#Making a single prediciton
import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r"D:\MBA - Balaji\Sem 3\Deep Learning\CNN Dataset\dataset\single_prediction\cat_or_dog_1.jpg",
                         target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)