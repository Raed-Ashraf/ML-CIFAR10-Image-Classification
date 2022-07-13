import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# merging training set
file_batch_1 = r'D:\Courses\Machine learning\Projects\02- Image Classification\cifar-10-batches-dataset\data_batch_1'
data_batch_1 = unpickle(file_batch_1)
file_batch_2 = r'D:\Courses\Machine learning\Projects\02- Image Classification\cifar-10-batches-dataset\data_batch_2'
data_batch_2 = unpickle(file_batch_2)
file_batch_3 = r'D:\Courses\Machine learning\Projects\02- Image Classification\cifar-10-batches-dataset\data_batch_3'
data_batch_3 = unpickle(file_batch_3)
file_batch_4 = r'D:\Courses\Machine learning\Projects\02- Image Classification\cifar-10-batches-dataset\data_batch_4'
data_batch_4 = unpickle(file_batch_4)
file_batch_5 = r'D:\Courses\Machine learning\Projects\02- Image Classification\cifar-10-batches-dataset\data_batch_5'
data_batch_5 = unpickle(file_batch_5)

x_train = np.concatenate([data_batch_1['data'],data_batch_2['data'],data_batch_3['data']
                        ,data_batch_4['data'],data_batch_5['data']], axis=0)
y_train = data_batch_1['labels']+data_batch_2['labels']+data_batch_3['labels']+data_batch_4['labels']+data_batch_5['labels']

file_test = r'D:\Courses\Machine learning\Projects\02- Image Classification\cifar-10-batches-dataset\test_batch'
test_batch = unpickle(file_test)

x_test = test_batch['data']
y_test = test_batch['labels']

# normalizing data input
x_train = x_train/255
x_test = x_test/255
# convert the labels to numpy array as the tensorflow model need
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# reshape the input
print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
# Reshape the whole image data
x_train = x_train.reshape(len(x_train),3,32,32)
x_test = x_test.reshape(len(x_test),3,32,32)
print("Shape after reshape and before transpose:", x_train.shape)
print("Shape after reshape and before transpose:", x_test.shape)
# Transpose the whole data
x_train = x_train.transpose(0,2,3,1)
x_test = x_test.transpose(0,2,3,1)
print("Shape after reshape and transpose:", x_train.shape)
print("Shape after reshape and transpose:", x_test.shape)


# visualizing our data
# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i])
#     plt.xlabel(class_names[y_train[i]])
#
# plt.show()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train, epochs=10, validation_data=(x_test,y_test))

loss, accuracy = model.evaluate(x_test, y_test)
print(f"loss: {loss}")
print(f"accuracy: {accuracy}")

model.save("image_classifier.model")
