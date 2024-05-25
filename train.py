# import pickle
#
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
# data_dict = pickle.load(open('./data.pickle', 'rb'))
#
# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])
#
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, shuffle = True, stratify = labels)
#
# model = RandomForestClassifier()
#
# model.fit(x_train, y_train)
#
# y_predict = model.predict(x_test)
#
# score = accuracy_score(y_predict, y_test)
#
# print('{}% of samples were classified correctly !'.format(score * 100))
#
# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()

import pickle
import numpy as np
from keras.utils import to_categorical
from keras.applications import VGG19
from keras.layers import Dense, Flatten
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load data
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)
x_data = data_dict['data']
y_data = data_dict['labels']

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_data))
y_data = to_categorical(y_data, num_classes)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Build VGG19 model
img_height, img_width, channels = x_data.shape[1:]
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_height, img_width, channels))
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generator
datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=0)
train_generator = datagen.flow(x_train, y_train, batch_size=32)
test_generator = datagen.flow(x_test, y_test, batch_size=32)

# Train the model
model.fit(train_generator, steps_per_epoch=len(x_train) // 32, epochs=10, validation_data=test_generator, validation_steps=len(x_test) // 32)

# Evaluate the model
scores = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy: {scores[1] * 100}%')

#
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()