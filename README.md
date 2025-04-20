# Face Mask Detection (Custom CNN Model)

This project builds and evaluates a deep learning model to detect whether a person is **wearing a mask** or **not**, based on images, videos, or real-time webcam input.

### Dataset:
- Source: [Kaggle Dataset by ashishjangra27](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
- Structure:
  - Train
  - Validation
  - Test
- Two Classes: `WithMask` and `WithoutMask`


## Steps and Code Overview:

### 1. **Data Extraction:**
- Loaded images and labels from train, validation, and test directories.
- Data stored into Pandas DataFrames.
- Labels were mapped:  
  - `WithMask` → `1`  
  - `WithoutMask` → `0`
  
### 2. **Data Visualization:**
- Random samples from the training set were plotted using `Matplotlib` to visualize the dataset.

### 3. **Data Generator:**
- A **Custom Data Generator** was built using `tf.keras.utils.Sequence`:
  - Loads images in batches.
  - Resizes them to `(224x224)`.
  - Normalizes pixel values between `0 and 1`.

### 4. **Model Architecture (Custom CNN):**
- Input: (224, 224, 3)
- Layers:
  - Conv2D → BatchNorm → MaxPooling
  - Conv2D → BatchNorm → MaxPooling
  - Conv2D → BatchNorm → MaxPooling
  - Conv2D → BatchNorm → MaxPooling
  - Conv2D → BatchNorm → GlobalMaxPooling
  - Dense → Dropout (0.4)
  - Dense → Dropout (0.4)
  - Output: Dense(1) with `sigmoid` activation.
- Weight Initialization: `he_normal`
- Regularization: Dropout layers to prevent overfitting.

### 5. **Model Compilation:**
- Optimizer: `Adam` with `Exponential Decay` Learning Rate Scheduler.
- Loss Function: `Binary Crossentropy`
- Metrics: `Accuracy`

### 6. **Training:**
- (Training code provided but commented out as model was already trained.)
- Debugging Callback prints intermediate loss and accuracy during training.
- Model weights are already saved and loaded (`ModelWeights.h5`).
- Achieved high accuracy and low loss on test data.

### 7. **Model Saving:**
- Saved Model: `Face_Mask_Model.h5`
- Saved Weights: `ModelWeights.h5`

### 8. **Evaluation:**
- Achieved:
   - Evaluated the model on the test set.
   - Final Results:
     - **Loss**: 0.0302
     - **Accuracy**: 0.9929

9. **Inference Options**:
   - **Image Inference**:
     - Predict mask/no-mask for a single image.
   - **Video Inference**:
     - Predict frame-by-frame from a saved video.
   - **Webcam Inference**:
     - Real-time mask detection using your webcam.
     - Uses **Haar Cascade** to detect faces and classify mask status.

10. **Additional Features**:
   - **Face Detection** in webcam mode using `haarcascade_frontalface_default.xml`.
   - Visualizations for random sample images.
   - Training vs Validation accuracy plotting function (commented but available).


## Inference Methods:

### 1. **Single Image Inference:**
- Load and predict a user-input image.
- Shows the prediction (`Mask` or `No Mask`) and confidence score.

### 2. **Video Inference:**
- User inputs a video file path.
- Processes frame-by-frame, shows real-time predictions.

### 3. **Webcam Inference:**
- Accesses laptop/computer webcam.
- Detects faces using **OpenCV Haar Cascade**.
- Predicts whether each detected face is wearing a mask or not in real-time.



## Important Files:
- `Face_Mask_Model.h5` → Full Model
- `ModelWeights.h5` → Model Weights
- Dataset Images (Train, Test, Validation)  



## Requirements:
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- OpenCV (cv2)

Install dependencies via:

```bash
pip install tensorflow numpy pandas matplotlib opencv-python
```



## Running Inference:

After running the script:
- Choose between:
  - `1` - Image Inference
  - `2` - Video Inference
  - `3` - Webcam Inference
  - `4` - Exit

Example:
```bash
Enter the type of Inference you want to do: 1
Enter the Image path: images/sample.jpg
```

## File Structure
```
├── Face Mask Dataset
│   ├── Train
│   ├── Validation
│   ├── Test
├── ModelWeights.h5
├── Face_Mask_Model.h5
├── your_script.py
```



# Final Notes:
- Custom CNN model was trained **from scratch**.
- High accuracy achieved without using any pretrained models.
- Supports **image**, **video**, and **real-time webcam** mask detection.
- Simple yet powerful architecture!


