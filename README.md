# Face_Emotion_Recognition


![image](https://github.com/AniketBembale08/Face_Emotion_Recognition/assets/121147984/00a3cd2f-b62f-4d07-b938-8208cc69f0b1)

## ⭐️Project Structure
Face_Emotion_Recognition/
  |-- README.md
  |-- requirements.txt
  |-- face_emotion.ipynb
  |-- image_test.py
  |-- realtime_test.py
  |-- models/
    |-- model.h5

  

## ⭐️Introduction
Facial Emotions Detection is an exciting computer vision project that aims to automatically detect and classify emotions expressed in human faces. This project utilizes deep learning techniques to recognize emotions such as happiness, sadness, anger, fear, surprise, disgust, and neutrality from facial expressions captured in images and real-time video streams.

## ⭐️Dataset
The project uses the Kaggle FER 2013 dataset, which contains labeled facial images for various emotions. The dataset has been preprocessed and split into training and testing sets to train and evaluate the emotion recognition model.

## ⭐️Installation
To run this project locally, follow these steps:
1. Clone the repository: `git clone https://github.com/AniketBembale08/Face_Emotion_Recognition.git`
2. Navigate to the project directory: `cd Face_Emotion_Recognition`
3. Install the required dependencies: `pip install -r requirements.txt`

## ⭐️Training
The model training is performed using a Jupyter Notebook (`training.ipynb`). This notebook includes the code for loading and preprocessing the dataset, building and training the deep learning model, and saving the trained model as `model.h5`.

## ⭐️Testing
To test the trained model on images, run `python image_test.py`. This script loads the saved model and performs emotion detection on sample images. The predicted emotions are displayed on the images.

To perform real-time emotion detection on a webcam feed, run `python realtime_test.py`. This script uses OpenCV to access the webcam and displays real-time emotion predictions on the video stream.

## ⭐️Model
The trained model (`model.h5`) is a Convolutional Neural Network (CNN) that has been trained on the Kaggle FER 2013 dataset. It achieves an accuracy of 80% on the test set.

## ⭐️Results
The project achieves impressive results in accurately recognizing emotions from facial expressions. The trained model can be further fine-tuned or integrated into various applications for real-world use.

![output](https://github.com/AniketBembale08/Face_Emotion_Recognition/assets/121147984/f76a32af-541f-4252-a112-0398533baa3d)


## ⭐️Contributing
Feel free to contribute to this project by submitting pull requests or opening issues. Happy face detection! 😄

## License
This project is licensed under the [MIT License](LICENSE).

