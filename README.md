# Cats-vs-Dogs-Prediction

A Convolutional Neural Network (CNN) model to classify images of cats and dogs using TensorFlow/Keras.
This project trains a deep learning model on the Cats vs. Dogs dataset and achieves high accuracy in classifying images.

📌 Dataset Information
Dataset: Cats vs. Dogs (from Kaggle or TensorFlow Datasets)
Total Images: 25,000 images (cats & dogs)
Image Format: 150x150 RGB
Classes: 2 (Cat 🐱 and Dog 🐶)

Key Features
Preprocessing & Normalization: Standardizes images for improved training.
CNN Model: Uses Conv2D, MaxPooling, and Dense layers.
Evaluation & Accuracy: Achieves high accuracy in classifying cats vs. dogs.
Visualization: Plots training loss, accuracy, and predictions.


Folder Structure- 

cats-vs-dogs/
┣ 📂 dataset/  
┃ ┣ 📂 train/  
┃ ┃ ┣ 📂 cats/   # Training images of cats  
┃ ┃ ┣ 📂 dogs/   # Training images of dogs  
┃ ┣ 📂 validation/  
┃ ┃ ┣ 📂 cats/   # Validation images of cats  
┃ ┃ ┣ 📂 dogs/   # Validation images of dogs  
┣ 📂 models/  
┃ ┗ 📜 model.h5  # Saved trained model  
┣ 📂 notebooks/  
┃ ┗ 📜 train_cnn.ipynb  # Jupyter Notebook for training  
┣ 📜 train.py  # Main script to train the CNN  
┣ 📜 predict.py  # Load model & predict images  
┣ 📜 README.md  # Project documentation  
┣ 📜 requirements.txt  # List of dependencies  
┗ 📂 test_images/  # Folder to store images for prediction  

📊 Results
✔️ Achieved 90%+ accuracy on validation data!

📌 Next Steps
🏆 Improve accuracy with data augmentation (RandomFlip, RandomRotation).
🚀 Deploy the model using Flask or Streamlit for real-world applications.
📈 Hyperparameter tuning (learning rate, optimizer adjustments).
