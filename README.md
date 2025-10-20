# 🤟 Sign Language Recognition System

A comprehensive **American Sign Language (ASL) recognition system** that uses **computer vision** and **deep learning** to detect and classify hand gestures in real-time. The project includes model training, desktop application, and web deployment capabilities.

## 🎯 **Project Overview**

This project recognizes **24 ASL letters** (A-Y, excluding J and Z which require motion) using a **Convolutional Neural Network (CNN)** trained on the Sign Language MNIST dataset. The system achieves **83%+ accuracy** and provides real-time prediction with hand detection bounding boxes.

## 📋 **Features**

- ✅ **Real-time ASL recognition** with webcam input
- ✅ **Hand detection and bounding boxes** using MediaPipe
- ✅ **Multiple interfaces**: Desktop app, localhost web app, and deployment-ready web app
- ✅ **High accuracy**: 95%+ recognition rate for most of the letters
- ✅ **Mobile-friendly**: Responsive web interface for phones and tablets
- ✅ **Live predictions**: Top-3 confidence scores with visual feedback
- ✅ **Cross-platform**: Works on Windows, Mac, Linux, mobile devices

## 🗂️ **Project Structure**

### **📊 Dataset & Training**
```
sign_mnist_train.csv          # Training dataset (27,455 samples)
sign_mnist_test.csv           # Test dataset (7,172 samples)
sign_language.ipynb           # Jupyter notebook for model training
cnn_sign_language_model.h5    # Trained CNN model (83%+ accuracy)
```

### **🖥️ Desktop Application**
```
opencv_demo.py                # Main desktop application with OpenCV
american_sign_language.PNG    # ASL alphabet reference image
amer_sign2.png               # Additional reference images
amer_sign3.png               # Additional reference images
```

### **🌐 Web Applications**
```
opencv_demo.py               # Python gui testing
opencv_demo_web.py           # Localhost web version (exact desktop logic)
app.py                       # Production-ready deployment version
```

### **📦 Deployment & Configuration**
```
requirements_deploy.txt      # Optimized dependencies for deployment
Procfile                    # Heroku deployment configuration
render.yaml                 # Render.com deployment configuration
DEPLOYMENT_GUIDE.md         # Comprehensive deployment instructions
```

### **📁 Additional**
```
webcam/                     # Webcam utilities and configurations
LICENSE                     # Project license
.gitignore                  # Git ignore rules
```

## 🚀 **Quick Start**

### **Prerequisites**
```bash
Python 3.8+
OpenCV 4.8+
TensorFlow 2.13+
MediaPipe 0.10+
```

### **Installation**
```bash
# Clone the repository
git clone https://github.com/SouriRishik/Sign-Language-Gesture-Analysis.git
cd Sign-Language-Gesture-Analysis

# Install dependencies
pip install opencv-python tensorflow mediapipe numpy pillow flask

# Run desktop application
python opencv_demo.py

# OR run web application
python app.py
# Then visit: http://localhost:5000
```

## 🎮 **Usage**

### **Desktop Application (`opencv_demo.py`)**
- **Real-time recognition** with webcam
- **Bounding box detection** around hands
- **Live confidence scores** and predictions
- **Keyboard controls** for settings

### **Web Application (`app.py`)**
- **Browser-based interface** for any device
- **Mobile-optimized** touch controls
- **Real-time streaming** and prediction
- **Shareable URL** for worldwide access

### **Model Training (`sign_language.ipynb`)**
- **Data preprocessing** and augmentation
- **CNN architecture** design and training
- **Performance evaluation** and visualization
- **Model export** for production use

## 🧠 **Model Architecture**

**Convolutional Neural Network (CNN)**
- **Input**: 28x28 grayscale images
- **Architecture**: Multiple Conv2D + MaxPooling layers
- **Output**: 24 classes (ASL letters A-Y)
- **Accuracy**: 83%+ on test dataset
- **Training**: Sign Language MNIST dataset

## 📊 **Dataset Information**

**Source**: [Kaggle Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- **Training samples**: 27,455
- **Test samples**: 7,172
- **Classes**: 24 (A-Y, excluding J and Z)
- **Format**: 28x28 pixel grayscale images
- **Labels**: Integer encoded (0-23)

**Supported Letters**: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y

## 🌐 **Deployment Options**

### **1. Local Deployment**
```bash
python app.py
# Access at: http://localhost:5000
```

### **2. Cloud Deployment (FREE)**

**Render.com** (Recommended)
```bash
# Push to GitHub, then deploy on Render.com
# Uses: requirements_deploy.txt and automatic detection
```

**Railway.app**
```bash
# Connect GitHub repo to Railway
# Uses: Procfile for configuration
```

**Heroku**
```bash
heroku create your-app-name
git push heroku main
```

See `DEPLOYMENT_GUIDE.md` for detailed instructions.

## 🛠️ **Technical Stack**

### **Computer Vision**
- **OpenCV**: Image processing and webcam handling
- **MediaPipe**: Hand detection and landmark extraction

### **Machine Learning**
- **TensorFlow/Keras**: CNN model training and inference
- **NumPy**: Numerical computations
- **PIL**: Image manipulation

### **Web Development**
- **Flask**: Web framework for deployment
- **HTML/CSS/JavaScript**: Frontend interface
- **Gunicorn**: WSGI server for production

### **Development Tools**
- **Jupyter Notebook**: Model development and training
- **Git**: Version control
- **Requirements.txt**: Dependency management

## 📈 **Performance Metrics**

- **Model Accuracy**: 83%+ on test dataset
- **Real-time Performance**: ~30 FPS processing
- **Confidence Threshold**: 50% minimum for predictions
- **Detection Speed**: <1 second per frame
- **Cross-platform Compatibility**: Windows, Mac, Linux, mobile

## 🎯 **Use Cases**

- **Accessibility tools** for deaf and hard-of-hearing individuals
- **Educational applications** for learning ASL
- **Communication aids** in healthcare and service industries
- **Research platform** for gesture recognition development
- **Mobile apps** for instant ASL translation

## 🔄 **Workflow**

1. **Data Collection**: Sign Language MNIST dataset
2. **Model Training**: CNN with data augmentation
3. **Desktop Development**: OpenCV real-time application
4. **Web Development**: Flask-based web interface
5. **Deployment**: Cloud platforms for worldwide access
6. **Testing**: Accuracy validation and performance optimization

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 **Acknowledgments**

- **Dataset**: [Kaggle Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- **MediaPipe**: Google's hand detection framework
- **TensorFlow**: Machine learning platform
- **OpenCV**: Computer vision library
- **ASL Community**: For inspiration and validation

---

### 🚀 **Ready to Deploy?**

Check out `DEPLOYMENT_GUIDE.md` for step-by-step instructions to make your ASL recognition system available worldwide! 🌍

**Live Demo**: Deploy and share your URL for anyone to try! 📱💻
