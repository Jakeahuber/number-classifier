import PIL.Image
from PIL import ImageEnhance
from neural_net import Neural_Net
import torch
from torchvision import transforms
import cv2
import tkinter as tk
import numpy as np
import torch.nn.functional as nnf
import torch.nn as nn
from tkinter import * 
import math
import imutils

import matplotlib.pyplot as plt

torch.set_printoptions(precision=10)

x = 0
y = 0
def get_coords(event):
    global x, y 
    x = event.x
    y = event.y

def draw_number(event):
    global x, y
    drawing_canvas.create_line((x, y, event.x, event.y), fill='white', width=6)
    x = event.x
    y = event.y

def guess_number():
    neural_net = Neural_Net()
    neural_net.load_state_dict(torch.load('model.pth'))

    save_drawing_canvas()
    image = load_image()
    output = neural_net(image)
    print(output)

    probability = nnf.softmax(output, dim = 1)
    print(probability)

    top_probabilities, top_labels = torch.topk(probability, 10)
    print(top_probabilities)
    print(top_labels)
    
    _, predicted = torch.max(output.data, 1) # the class with the highest energy is what we choose as prediction
    result_label.config(text=f'This number is a {predicted.numpy()[0]}')

# As far as I'm aware, OpenCV does not support .ps files. This is why I used Pillow Instead. 
def save_drawing_canvas():
    drawing_canvas.postscript(file='drawn_image.ps')
    ps_image = PIL.Image.open('drawn_image.ps')
    ps_image.save('drawn_image.png')

    # Resizes image
    png_img = PIL.Image.open('drawn_image.png')
    png_img.save('drawn_image.png')

def load_image():
    image = cv2.imread('drawn_image.png', cv2.IMREAD_GRAYSCALE)
    _, threshold = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY) # Convert grayscale image to binary

    x,y,w,h = cv2.boundingRect(threshold)
    cropped_threshold = threshold[y:y+h, x:x+w]
    plt.imshow(cv2.cvtColor(cropped_threshold, cv2.COLOR_BGR2RGB))
    
    # When cropped, the image becomes grayscaled instead of binary
    if w > h:
        cropped_threshold = imutils.resize(cropped_threshold, width=20)
    else:
        cropped_threshold = imutils.resize(cropped_threshold, height=20)

    h, w = cropped_threshold.shape

    # Create a 28x28 all black background image, and paste the 20x20 drawing in the center of the background image.
    background_image = np.zeros((28, 28), dtype="uint8")
    offset_x = round((28 - w) / 2)
    offset_y = round((28 - h) / 2)
    background_image[offset_y:offset_y + h, offset_x:offset_x + w] = cropped_threshold 

    # Display image for testing purposes
    plt.imshow(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))
    plt.show()

    # Convert to tensor and return 
    background_image = torch.Tensor(background_image)
    return background_image.view(-1, 28*28).requires_grad_()

def clear_drawing_canvas():
    drawing_canvas.delete('all')
    create_drawing_canvas()

def create_drawing_canvas():
    drawing_canvas.bind("<Button-1>", get_coords)
    drawing_canvas.bind("<B1-Motion>", draw_number)
    drawing_canvas.create_rectangle(0, 0, 400, 400, fill='black')
    outer_canvas.create_window(250, 250, window=drawing_canvas)

root = Tk()
outer_canvas = Canvas(root, width=500, height=500)
outer_canvas.pack()

drawing_canvas = Canvas(outer_canvas, width=200, height=200)
create_drawing_canvas()

# Add buttons and text to the outer canvas
btn = Button(root, text='Guess Number', command=guess_number)
btn.place(x=200, y=425)

btn = Button(root, text='Clear Drawing', command=clear_drawing_canvas)
btn.place(x=200, y=450)

label = Label(text="Number classifier")
label.place(x=200,y=50)  

result_label = Label(text="")
result_label.place(x=200, y=75)

root.mainloop()
