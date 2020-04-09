# coding: utf-8

# In[4]:


from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

# In[8]:


cnnModel = Sequential


# In[17]:


def firstModel(model):
    #  Input Data ( First Hidden Layer)
    model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #     Second Hidden Layer
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #     Third Hidden Layer
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #     Flaten : use for output
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))


# In[18]:


def secondModel(model):
    # Convolution
    model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 3), activation='relu'))

    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening (use for output) , convert in 1D
    model.add(Flatten())

    # Full connection
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))


# In[ ]:


def thirdModel(model):


# In[19]:


def compiliModelForBinary(model, lossFunction, optumizer):
    model.complie(loss=lossFunction, optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


#     model.complie(loss = lossFunction, optimizer='rmsprop', metrics='accuracy')


# In[20]:


def compileModelForMultiple(model):
    # compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[26]:


def firstFit():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
    model.save_weights('cat_dog_model_128.h5')  # always save your weights after training or during training


# In[25]:


def secondFit():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    # test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory('train',
                                                     target_size=(50, 50),
                                                     batch_size=32,
                                                     class_mode='binary')
    model.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_steps=2000)


# In[28]:


def thirdFit(model):
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('first_try.h5')


# In[22]:


model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=800 // batch_size)
model.save_weights('cat_dog_model_128.h5')  # always save your weights after training or during training


# In[24]:


# **This is the augmentation configuration we will use for training and validation**
def dataGenForTrain():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    return train_datagen


def dataGenForValidation():
    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    return val_datagen


# In[29]:


def label_img(img):
    word_label = img.split('.')[-3]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'cat':
        return [1, 0]
    #                             [no cat, very doggo]
    elif word_label == 'dog':
        return [0, 1]


# In[30]:


TRAIN_DATASET = 'train'
TEST_DATASET = 'test1'
IMG_SIZE = 50


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DATASET)):
        label = label_img(img)
        path = os.path.join(TRAIN_DATASET, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    # np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DATASET)):
        path = os.path.join(TEST_DATASET, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    # np.save('test_data.npy', testing_data)
    return testing_data


# In[31]:


# **Now the images have to be represented in numbers. For this, using the openCV library read and resize the image.  **

# **Generate labels for the supervised learning set.**

# **Below is the helper function to do so.**

def prepare_data(list_of_images):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """
    x = []  # images as arrays
    y = []  # labels

    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image), (img_width, img_height), interpolation=cv2.INTER_CUBIC))

    for i in list_of_images:
        if 'dog' in i:
            y.append(1)
        elif 'cat' in i:
            y.append(0)
        # else:
        # print('neither cat nor dog name present in images')

    return x, y


# In[33]:


# **Generate X and Y using the helper function above**

# **Since K.image_data_format() is channel_last,  input_shape to the first keras layer will be (img_width, img_height, 3). 
# '3' since it is a color image**

X, Y = prepare_data(train_images_dogs_cats)
print(K.image_data_format())

# **Split the data set containing 2600 images into 2 parts, training set and validation set. Later,
# you will see that accuracy and loss on the validation set will also be reported 
# while fitting the model using training set.**

# First split the data in two sets, 80% for training, 20% for Val/Test_Amat)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=1)

# Or (get input and output column data) : get feature and label
# **Time to predict classification using the model on the tensorbroad_pb_android set.**

# **Generate X_test and Y_test**

X_test, Y_test = prepare_data(test_images_dogs_cats)  # Y_test in this case will be []
