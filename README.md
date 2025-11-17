

#  **C2_3_Emotion-based_music_recommendation_system**

### *Facial Emotion Recognition using CNN + AIML + Full-Stack Deployment*

---

##  **Project Overview**

This project detects human emotions from facial expressions using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset and recommends songs based on the detected emotion. The system integrates **computer vision**, **deep learning**, **backend deployment**, and **frontend UI** to create a real-time emotion-based recommendation system.

Users can **upload or capture a picture**, and the system:

1. Detects the face
2. Predicts the emotion
3. Loads the relevant CSV file
4. Recommends songs based on user mood

---

##  Dataset — FER-2013

* Total images: **35,887**
* Training: **28,709**
* Validation/Test: **7,178**
* Image size: **48 × 48 (grayscale)**
* Emotions:

  * Angry
  * Disgust
  * Fear
  * Happy
  * Neutral
  * Sad
  * Surprise

---

## CNN Model Architecture

* **Conv2D layers:** 32, 64, 128 filters
* **Activation:** ReLU
* **Pooling:** MaxPooling2D
* **Regularization:** Dropout
* **Dense layer:** 1024 neurons
* **Output:** Softmax (7 classes)

### Training Details

* Optimizer: **Adam**
* Loss: **Categorical Crossentropy**
* Epochs: **40–75**
* Accuracy Achieved:

  * **Training:** ~98–99%
  * **Validation:** ~70–85%

---

##  System Workflow

1. User uploads/captures an image
2. OpenCV detects face
3. Image is resized, grayscaled, normalized
4. CNN model predicts emotion
5. Corresponding CSV file is loaded
6. 5 songs from that emotion dataset are recommended
7. UI displays emotion + songs

---

##  Technologies Used

### AI / ML:

* TensorFlow/Keras
* CNN
* ImageDataGenerator
* Softmax classifier

### Computer Vision:

* OpenCV

### Backend:

* Flask
* Pandas
* NumPy

### Frontend:

* HTML
* CSS
* JavaScript

---

##  Project Structure

```
C2_3_Emotion-based_music_recommendation_system/
│
├── backend/
│   ├── app.py
│   ├── model.h5
│   ├── haarcascade_frontalface.xml
│   ├── csv_files/
│       ├── happy.csv
│       ├── sad.csv
│       ├── angry.csv
│       ├── fear.csv
│       ├── neutral.csv
│       └── surprise.csv
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│
├── README.md
└── requirements.txt
```

---

##  Running the Project

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Start backend:

```bash
python app.py
```

### Run frontend:

Just open **index.html** in any browser.

---

##  Results

* Real-time emotion detection (< 300 ms)
* Accurate emotion prediction
* Personalized music recommendations
* Smooth UI/Backend interaction

---

##  Unique Features

* End-to-end AIML + Full-stack integration
* Real-time, practical emotion detection
* Emotion-based music personalization
* Uses **custom-trained CNN model (.h5)**
* 6–7 CSV files for emotion-specific song datasets

---

##  Future Scope

* Spotify/Youtube Music API integration
* Real-time live video emotion analysis
* Transfer learning with EfficientNet/MobileNet
* Advanced UI/UX

---

##  License

This project is for academic and research purposes.

---


