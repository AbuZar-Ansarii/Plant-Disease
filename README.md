# Plant Disease Detection System 🌱🔍

A real-time plant disease detection system using OpenCV and deep learning that identifies diseases from live camera feed and provides treatment suggestions via Google search.

# Project Demo 
![Screenshot 2025-05-26 121058](https://github.com/user-attachments/assets/c7350949-6ce3-41e5-bfec-5ebb5a0e0bd6)

![Screenshot 2025-05-26 121143](https://github.com/user-attachments/assets/18ff40fb-6237-42cd-be0d-dcff22ff1706)

# Search Result
Search Results for Treatment Information:
/search?num=7
https://www.gardeningknowhow.com/edible/vegetables/pepper/common-pepper-plant-problems.htm
https://www.hb-p.com/article/common-pepper-diseases-and-how-to-deal-with-them/      
https://www.sciencedirect.com/science/article/pii/S1517838216307663
https://plantvillage.psu.edu/topics/pepper-bell/infos

## Features ✨

- **Real-time detection** from camera feed or video stream
- **Target area framing** for precise plant placement
- **Disease classification** using a pre-trained CNN model
- **Confidence scoring** for each prediction
- **Automatic Google search** for treatment options

## Technologies Used 🛠️

- Python 3.x
- OpenCV (for image processing and camera feed)
- TensorFlow/Keras (for deep learning model)
- scikit-learn (for model training - if applicable)
- webbrowser (for Google search integration)

## Installation Guide 💻

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/AbuZar-Ansarii/Plant-Disease.git
cd plant-disease-detection
