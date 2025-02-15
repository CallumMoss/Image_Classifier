from train import FCNN
import torch
from PIL import Image, ImageOps, ImageGrab
from torchvision import transforms
import tkinter as tk
import torch.nn.functional as F
import math

def preprocess_canvas_image(canvas):
    # Capture the canvas
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    img = ImageGrab.grab(bbox=(x, y, x1, y1)).convert('L')  # Convert to grayscale

    # Invert colors (MNIST has white digits on black background)
    img = ImageOps.invert(img)

    # Resize while keeping aspect ratio and adding padding if needed
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Normalize pixel values to match MNIST (mean=0.5, std=0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img_tensor = transform(img)
    
    return img_tensor

# Prediction function
def predict_digit(model, image_tensor):
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    output = model(image_tensor)
    # Apply softmax to convert logits in the class dimension to probabilities
    probabilities = F.softmax(output, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    print_probabilities(probabilities)
    print(f"Predicted Label: {predicted_label}")

def print_probabilities(probabilities):
    print(f"Probability Distribution:")
    raw_probabilities = probabilities.detach()[0]  # Get the probabilities without the extra info for PyTorch
    for i, prob in enumerate(raw_probabilities):
        print(f"Class {i}: {round(prob.item() * 100, 2)}%")


# GUI for drawing digit
def draw(event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)

def predict_from_canvas():
    image_tensor = preprocess_canvas_image(canvas)
    predict_digit(model, image_tensor)

def clear_canvas():
    canvas.delete("all")

if __name__ == "__main__":
    # Load trained model
    PATH = "model/image_classifier.pt"
    model = FCNN()
    model.load_state_dict(torch.load(PATH))
    model.eval()

    # GUI setup
    root = tk.Tk()
    root.title("Draw a Digit")
    canvas = tk.Canvas(root, width=200, height=200, bg='white')
    canvas.pack()

    canvas.bind("<B1-Motion>", draw)

    button_frame = tk.Frame(root)
    button_frame.pack()

    predict_button = tk.Button(button_frame, text="Predict", command=predict_from_canvas)
    predict_button.pack(side=tk.LEFT)

    clear_button = tk.Button(button_frame, text="Clear", command=clear_canvas)
    clear_button.pack(side=tk.RIGHT)

    root.mainloop()