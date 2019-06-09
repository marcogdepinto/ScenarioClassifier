import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

model = keras.models.load_model('./model/scenarioclassifier_cat_new.h5')

print(model.summary())

test_data_dir = './dataset/test/'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
 test_data_dir,
 target_size=(200, 200),
 batch_size=32,
 class_mode='categorical',
 shuffle=False)

pred = model.predict_generator(test_generator,steps = 150/32)

predicted_classes = np.argmax(pred, axis=1)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print(classification_report(true_classes, predicted_classes, target_names=class_labels))
print(confusion_matrix(true_classes, predicted_classes))
