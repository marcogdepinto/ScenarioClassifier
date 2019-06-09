from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import os

callback = [EarlyStopping(monitor='acc', patience=1, verbose=1)]

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape=(200, 200, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(6, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Printing model

plot_model(classifier, to_file='model.png', show_shapes=True)

# Processing images

train_datagen = ImageDataGenerator(rescale=1./255,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True)

train_data_dir = './dataset/training/'
test_data_dir = './dataset/test/'

test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(train_data_dir,
 target_size=(200, 200),
 batch_size=32,
 class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_data_dir,
 target_size=(200, 200),
 batch_size=32,
 class_mode='categorical')

classifier.fit_generator(training_set,
 validation_data=test_set,
 steps_per_epoch=500,
 epochs=5,
 validation_steps=100,
 callbacks=callback)

model_name = 'scenarioclassifier_cat_new.h5'
save_dir = './model/'
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
classifier.save(model_path)
print('Saved trained model at %s ' % model_path)