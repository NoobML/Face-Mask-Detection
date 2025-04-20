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


train_directory = 'Face Mask Dataset\Train'
validation_directory = 'Face Mask Dataset\Validation'
test_directory = 'Face Mask Dataset\Test'

def extract_data_from_directory(directory_path):
    paths = []
    labels = []

    subfolders = os.listdir(directory_path)
    for subfolder in subfolders:
        subfolder_path = os.path.join(directory_path, subfolder)
        image_files = os.listdir(subfolder_path)

        for image in image_files:
            image_path = os.path.join(subfolder_path, image)
            paths.append(image_path)
            labels.append(subfolder)


    paths_series = pd.Series(paths, name='filePath')
    labels_series = pd.Series(labels, name='Label')
    data = pd.concat([paths_series, labels_series], axis=1)

    return data


def extract_all_data():
    train_data = extract_data_from_directory(train_directory)
    validation_data = extract_data_from_directory(validation_directory)
    test_data = extract_data_from_directory(test_directory)

    return train_data, validation_data, test_data

train_data, validation_data, test_data = extract_all_data()


random_images = train_data.sample(n=6, random_state=0)
fig, ax = plt.subplots(3, 2, figsize=(8, 7))

for i, (idx, row) in enumerate(random_images.iterrows()):
    label = row['Label']
    img_path = row['filePath']
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    row_idx = i // 2
    col_idx = i % 2

    ax[row_idx, col_idx].imshow(img)
    ax[row_idx, col_idx].axis('off')
    ax[row_idx, col_idx].set_title(label)
# plt.show()

train_data['Label'] = train_data['Label'].map({'WithMask': 1, 'WithoutMask': 0})
validation_data['Label'] = validation_data['Label'].map({'WithMask': 1, 'WithoutMask': 0})
test_data['Label'] = test_data['Label'].map({'WithMask': 1, 'WithoutMask': 0})


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=32, image_size=(224, 224)):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.dataframe.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]

        images = np.array([
            img_to_array(load_img(filepath, target_size=self.image_size, color_mode='rgb')) / 255.0
            for filepath in batch_df['filePath']
        ])

        labels = np.array(batch_df['Label'].astype(np.float32))

        return images, labels



seed = 42

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

model.load_weights('ModelWeights.h5')

steps_per_epoch = len(train_data) // 32
validation_steps = len(validation_data) // 32


lr_schedule = ExponentialDecay(
    initial_learning_rate=0.0005,  # Drastically reduced from 0.1
    decay_steps=steps_per_epoch * 5,
    decay_rate=0.1,
    staircase=True
)

model.compile(optimizer=Adam(learning_rate=lr_schedule),
             loss='binary_crossentropy',
             metrics=['accuracy'])



class DebugCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if batch % 10 == 0:  # Print every 10 batches
            print(f"Batch {batch}, loss: {logs['loss']:.4f}, acc: {logs['accuracy']:.4f}")



# history = model.fit(train_generator, epochs=200, validation_data=validation_generator,
#                     steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
#                     callbacks=[DebugCallback()])


model.save('Face_Mask_Model.h5')
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

# plot_accuracy(history)


def Inference_Image():
    print("Running Image Inference...")
    image_path = input('Enter the Image path: ')

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image.")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    plt.imshow(image)
    plt.axis('off')
    plt.show()

    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)
    print(f'Prediction result', pred)

    if pred > 0.5:
        print('The person has worn a mask')
    else:
        print('The person has NOT worn a mask')


def video_inference():
    print("Running Video Inference...")
    video_path = input('Enter the Video path: ')
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (224, 224))
        frame_input = resized_frame / 255.0
        frame_input = np.expand_dims(frame_input, axis=0)

        predictions = model.predict(frame_input, verbose=0)

        label = 'Mask' if predictions[0][0] > 0.5 else 'No Mask'
        color = (0, 0, 255) if label == 'No Mask' else (0, 255, 0)

        cv2.putText(display_frame, f"{label}: {predictions[0][0]:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('Video Frame', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def webcam_inference():
    print("Running Webcam Inference...")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = frame[y:y + h, x:x + w]

            resized = cv2.resize(face, (224, 224))
            normalized = resized / 255.0
            input_frame = np.expand_dims(normalized, axis=0)

            preds = model.predict(input_frame)
            label = 'Mask' if preds[0][0] > 0.5 else 'No Mask'

            color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('Webcam Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




def Select_type_of_Inference(option):
    if option == 1:
        Inference_Image()
    elif option == 2:
        video_inference()
    elif option == 3:
        webcam_inference()
    elif option == 4:
        exit()
    else:
        print("Invalid option. Please select a valid option (1, 2, or 3).")


def start_Inference():
    print('1- Inference Image')
    print('2- Video Inference')
    print('3- Video Inference')
    print('4- Exit')

    while True:
        try:
            option = int(input('Enter the type of Inference you want to do: '))
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 3.")
            continue

        Select_type_of_Inference(option)
        flag = input('Do you want to continue(Y) or exit(N)? : ').strip().lower()
        if flag == 'n':
            print("Exiting....")
            break

start_Inference()