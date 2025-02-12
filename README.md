# **Handwritten Digit Recognition**  
This is a personal project that allows you to draw a digit (0-9) and have a trained machine learning model predict the number (assuming honest inputs).  

The model is trained on the **MNIST dataset**, which consists of **60,000 images** of handwritten digits.  

**Dataset:** [MNIST Dataset on Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)  

---

## **Optimized Hyperparameters**  
After running **20 trials** and training for **40 epochs** using `find_best_parameters.py`, the best parameters were found to be:  

- **Learning Rate:** `0.001065741156220378`  
- **Batch Size:** `128`  
- **Epochs:** `40` (not optimized, as performance plateaus beyond 40)  

These parameters were chosen to balance accuracy and training efficiency while preventing overfitting.  

---

## **Installation & Usage**  

### **1. Install Dependencies**  
Ensure you have Python installed, then install the required dependencies:  
```bash
pip install -r requirements.txt

### **2. Optimising Hyperparameters (Optional)**
Run the optimization script to find the best hyper parameters
```bash
python find_best_parameters.py
```

### **3. Training (Optional)**
Train the model on desired parameters
```bash
python train.py
```

### **4. Prediction**
Run the model. This will open a small canvas to draw a number on. Simply click predict to see the result.
```bash
python use_model.py
```