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
```
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

---

## **How The Model Performs Inference**
After allowing our user to draw their number and interact with the predict button, we must now perform the inference.
The following sections thoroughly explain the pipeline for the FCNN. Typically a CNN would be used for image inference, however as the inputs are so simple, we can skip convolutions, pooling and flatenning, and head straight to the FCNN phase. If you understand neural networks well already, I recommend simply reading the headings to understand the pipeline.

### **1. Load the model**
We load our pretrained model state from "model/image_classifier.pt" and apply it to a newly initialised model from the local FCNN class. Next we enable evaluation mode. This stops dropout occuring which is typically used to zero out some neurons during training to prevent overfitting.

### **2. Gather the input**
Now that the model is loaded, we must gather the input. This involves using the tkinter library to create a canvas for the user to draw their digit onto and creating buttons to either ask the model for a prediction, or to clear the canvas so they can try again.

### **3. Preprocess the input**
Once the user is ready, they will interact with the predict button. This calls the predict_from_canvas function which preprocesses the image and calls the prediction_digit function. Convert our input to match MNIST data.

#### **3.1 Convert to grayscale**
The MNIST dataset consists images that have been converted to grayscale. This is because such we can save memory by only story 1 channel rather than 3 channels if we were using RGB. It stores values for pixels in terms of brightness. As our image is just black and white, no information should be lost.
Therefore, we also convert our input to grayscale using the function: .convert('L').

#### **3.2 Invert the image**
The MNIST dataset consists of white digits on black backgrounds. However, we capture black digits on white backgrounds. This is because it is much simpler to implement. Therefore, we simply invert the image to match the form of the MNIST data. We use the function: .invert(img)


#### **3.3 Resizing**
The MNIST dataset consists of (1,28,28) images. Simply put, the 1 represents the number of channels which we discussed earlier, and the images themselves are 28x28 pixels. It would be difficult for people to draw numbers accurately with only 28 pixels, so we allow a 200x200 input.
We scale down the image to be 28x28 using Lanczos resampling to maintain image quality.

#### **3.4 Converting PIL to Tensor**
We apply a chain of transformations to finish our preprocessing. We convert the image from PIL format to Tensor format. In other words, we normalise pixel values from 0 - 255 to 0.0 - 1.0. This conversion also changes the image vector to be in the format (Channels, Height, Width) from (Height, Width Channels). This is because PyTorch expect a tensor to be in CHW format. For reference, a tensor can be thought of simply as a general term for an array with n dimensions. A scalar is a 0D tensor (just a single number), a vector is a 1D tensor (1D array), a matrix is a 2D tensor etc. Any tensor with 3 or more dimensions do not have their own word.

#### **3.5 Normalising the Tensor**
Finally, we normalise the pixel values to match the MNIST data. We set the mean and standard deviation to 0.5, therefore converting our pixel value range from 0 to 1 -> -1 to 1. Formula used: normalized value = (original value − 0.5)/0.5.
A white pixel should be 1, and a black pixel -1.
Because our image are grayscale, we only use 1 channel. This means only providing a single mean and standard deviation.

### **4. Prediction**
We are now ready to perform inference.

#### **4.1 Adding a batch dimension**
Because PyTorch expects a batch of inputs, and we are infering one image, we must add a batch dimension using unsqueeze(0). This is a simpler approach than reconfiguring the model to accept single images. All this does is add an extra dimension to our tensor. We go from CHW to BCHW, in other words: (B=1,C=1,H=28,W=28), B is the batch size, C is the number of channels, H is the image height in pixels and W is the image width in pixels.

#### **4.2 Passing the tensor into the model**
We pass our tensor into the model which performs forward propagation through its layers to produce an output tensor of shape [1, 10], the batch size is 1 and an output is produced for each of the 10 possible classes the digit could be, 0 to 9. The output is 10 numbers between 0 and 1, that sum to 0 and 1 which represent the probability of the image belonging to the associated class.

#### **4.3 Extract and Output the Predicted Label**
We now use torch.argmax to extract the index of the highest value in the output tensor across the class dimension. We access dim=1, which tells it to consider the second dimension which corresponds to the classes dimension. We then get the class name itself from the index using .item(). Finally, we simply print the predicted label.

---

### **How the model is optimised and trained**
(TBD)
CrossEntropyLoss automatically applies softmax

---

## **Improvements**
In the future I would like to improve the preprocessing to account for intentional misleading inputs, such as digits drawn close to the edge of the canvas.