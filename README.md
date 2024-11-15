# Hand-Gesture-recognition

# Hand Gesture Recognition Using Convolutional Neural Networks  

## Project Overview  
This project demonstrates the development of a **Hand Gesture Recognition System** using a **Convolutional Neural Network (CNN)**. The model is trained on the **LeapGestRecog Dataset**, which contains various hand gesture images, to classify different gestures effectively.  

## Dataset  
The dataset used is the **LeapGestRecog Dataset** with gesture images organized in folders based on participants and gesture types. The images are grayscale and resized to **64x64 pixels** for uniformity.  

- **Dataset Path:** `archive/leapGestRecog/leapGestRecog`
- **Classes:** Hand gesture categories  
- **Preprocessing:** Normalized pixel values and one-hot encoded labels.  

---

## Model Architecture  

The CNN model comprises:  
1. **Convolution Layers**: Extract spatial features with ReLU activation.  
2. **MaxPooling Layers**: Reduce spatial dimensions and computational cost.  
3. **Flattening Layer**: Converts feature maps into a 1D array.  
4. **Fully Connected Dense Layers**: Learn complex representations.  
5. **Dropout Layer**: Prevents overfitting.  
6. **Output Layer**: Softmax activation for multi-class classification.  

---

## Key Libraries Used  
- **NumPy**: Data manipulation and processing.  
- **OpenCV**: Image processing.  
- **Keras**: Building and training the CNN model.  
- **Matplotlib**: Visualizing accuracy, loss, and sample predictions.  
- **Scikit-learn**: Data splitting and label encoding.  

---

## Steps to Run the Code  
1. **Install Prerequisites**:  
   - Ensure Python is installed along with the required libraries. Install dependencies using:  
     ```bash
     pip install numpy opencv-python matplotlib scikit-learn keras tensorflow
     ```  

2. **Set Dataset Path**:  
   Place the **LeapGestRecog Dataset** in the `archive/leapGestRecog` directory.  

3. **Run the Code**:  
   Execute the script to preprocess the dataset, train the CNN model, and evaluate its performance.  

4. **Visualizations**:  
   - Training and validation accuracy/loss graphs.  
   - Sample gesture images with their predicted labels.  

---

## Results  
- **Accuracy**: The model achieves competitive accuracy on training and validation sets.  
- **Confusion Matrix and Classification Report**: Evaluate the performance of the model on unseen data.  
- **Sample Predictions**: Displays gesture images with predicted labels for visual verification.  
