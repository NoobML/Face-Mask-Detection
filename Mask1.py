import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Concatenate, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import os
import matplotlib.pyplot as plt
import cv2



# # Given paths
train_directory = r'C:\Users\Hacx\Documents\ML\Kaggle\Mask Detection\Face Mask Dataset\Train'
validation_directory = r'C:\Users\Hacx\Documents\ML\Kaggle\Mask Detection\Face Mask Dataset\Validation'
test_directory = r'C:\Users\Hacx\Documents\ML\Kaggle\Mask Detection\Face Mask Dataset\Test'

def extract_data_from_directory(directory_path):
    """
    This function extracts image paths and their corresponding labels from a given directory.
    It assumes that the directory contains subfolders, and the subfolder name is the label.
    """
    paths = []
    labels = []

    # Get subfolders (which are the labels)
    subfolders = os.listdir(directory_path)
    for subfolder in subfolders:
        subfolder_path = os.path.join(directory_path, subfolder)  # Full path to the subfolder
        image_files = os.listdir(subfolder_path)  # List all image files in the subfolder

        for image in image_files:
            image_path = os.path.join(subfolder_path, image)  # Full path to each image
            paths.append(image_path)
            labels.append(subfolder)  # Folder name is the label (e.g., 'withMask', 'withoutMask')

    # Create a DataFrame with image paths and labels
    paths_series = pd.Series(paths, name='filePath')
    labels_series = pd.Series(labels, name='Label')
    data = pd.concat([paths_series, labels_series], axis=1)

    return data


def extract_all_data():
    """
    This function extracts train, validation, and test data using the generalized function.
    """
    # Extract data for all three datasets
    train_data = extract_data_from_directory(train_directory)
    validation_data = extract_data_from_directory(validation_directory)
    test_data = extract_data_from_directory(test_directory)

    return train_data, validation_data, test_data

train_data, validation_data, test_data = extract_all_data()

# # lets look at few images in the dataset

random_images = train_data.sample(n=6, random_state=0)
fig, ax = plt.subplots(3, 2, figsize=(8, 7))

for i, (idx, row) in enumerate(random_images.iterrows()):
    label = row['Label']
    img_path = row['filePath']
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # h, w, c = img.shape
    # print(h, w, c) 115 115 3, 224 224 3, 97 97 3


    # Convert 1D index (i) to 2D row and column index for ax
    row_idx = i // 2  # Determine row index
    col_idx = i % 2  # Determine column index

    ax[row_idx, col_idx].imshow(img)
    ax[row_idx, col_idx].axis('off')
    ax[row_idx, col_idx].set_title(label)
# plt.show()

train_data['Label'] = train_data['Label'].map({'WithMask': 1, 'WithoutMask': 0})
validation_data['Label'] = validation_data['Label'].map({'WithMask': 1, 'WithoutMask': 0})
test_data['Label'] = test_data['Label'].map({'WithMask': 1, 'WithoutMask': 0})



# # Train Data Generator with Augmentation
# train_datagen = ImageDataGenerator(
#     rescale = 1./255, # Normalize images [0,1]
#     rotation_range = 30, # Random Rotations
#     width_shift_range=0.2, # Random horizontal shift
#     height_shift_range=0.2, # Random vertical shift
#     shear_range= 0.2, # random shear transformation
#     zoom_range=0.2, # Random zoom
#     horizontal_flip=True, #Randomly flip images
#     fill_mode='nearest' #how to fill newly created pixels
# )


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=32, image_size=(224, 224)):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        # This is the number of batches per epoch
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.dataframe.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]

        # Load and normalize images
        images = np.array([
            img_to_array(load_img(filepath, target_size=self.image_size, color_mode='rgb')) / 255.0
            for filepath in batch_df['filePath']
        ])

        # Convert labels to a NumPy array with appropriate shape and type
        labels = np.array(batch_df['Label'].astype(np.float32))  # Ensure labels are NumPy array and have correct dtype

        # Reshape labels to match the output shape (e.g., (batch_size, 1) for binary classification)
        return images, labels  # For binary classification, reshape to (batch_size, 1)


# Set a fixed random seed for reproducibility
seed = 42  # You can use any integer value here

train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)
validation_data = validation_data.sample(frac=1, random_state=seed).reset_index(drop=True)
test_data = test_data.sample(frac=1, random_state=seed).reset_index(drop=True)

train_generator = CustomDataGenerator(train_data)
validation_generator = CustomDataGenerator(validation_data)
test_generator = CustomDataGenerator(test_data)



def create_model():
    image_input = Input(shape=(224, 224, 3), name='image')
    x = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal')(image_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling2D()(x)

    x = Dense(32, activation='relu',  kernel_initializer='he_normal')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.4)(x)

    z = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=image_input, outputs=z)
    return model

model = create_model()

# Load the model weights
# model.load_weights('ModelWeights.keras')

steps_per_epoch = len(train_data) // 32
validation_steps = len(validation_data) // 32


lr_schedule = ExponentialDecay(
    initial_learning_rate=0.0005,  # Drastically reduced from 0.1
    decay_steps=steps_per_epoch * 5,
    decay_rate=0.1,
    staircase=True
)
# Or even simpler:
model.compile(optimizer=Adam(learning_rate=lr_schedule),
             loss='binary_crossentropy',
             metrics=['accuracy'])


# Add this callback to monitor training
class DebugCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 10 == 0:  # Print every 10 batches
            print(f"Batch {batch}, loss: {logs['loss']:.4f}, acc: {logs['accuracy']:.4f}")


# Use it in model.fit()
history = model.fit(train_generator, epochs=200, validation_data=validation_generator,
                    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                    callbacks=[DebugCallback()])


model.save('Model.h5')
model.save_weights('ModelWeights.h5')

steps_per_epoch = len(test_data) // 32
loss, accuracy = model.evaluate(test_generator, steps=steps_per_epoch)
print(f'Loss: {loss}, Accuracy: {accuracy}')



def plot_accuracy(history):
    epochs = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(12, 10))
    plt.plot(epochs, history.history['accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r*-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation accuracy')
    plt.legend()
    plt.show()

plot_accuracy(history)


