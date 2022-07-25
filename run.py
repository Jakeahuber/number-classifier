import PIL.Image
from PIL import ImageEnhance
from neural_net import Net
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
from numpy import newaxis

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

def clear_drawing_canvas():
    drawing_canvas.delete('all')
    create_drawing_canvas()

def create_drawing_canvas():
    drawing_canvas.bind("<Button-1>", get_coords)
    drawing_canvas.bind("<B1-Motion>", draw_number)
    drawing_canvas.create_rectangle(0, 0, 400, 400, fill='black')
    outer_canvas.create_window(250, 250, window=drawing_canvas)

def save_drawing_canvas():
    drawing_canvas.postscript(file='drawn_image.ps')
    ps_image = PIL.Image.open('drawn_image.ps')
    ps_image.save('drawn_image.png')
    center_image()

def center_image():
    image = cv2.imread('drawn_image.png', cv2.IMREAD_GRAYSCALE)
    _, threshold = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY) # Convert grayscale image to binary

    x,y,w,h = cv2.boundingRect(threshold)
    cropped_threshold = threshold[y:y+h, x:x+w]
    plt.imshow(cv2.cvtColor(cropped_threshold, cv2.COLOR_BGR2RGB))
    
    # When cropped, the image becomes grayscaled instead of binary
    if w > h:
        cropped_threshold = imutils.resize(cropped_threshold, width=200)
    else:
        cropped_threshold = imutils.resize(cropped_threshold, height=200)

    h, w = cropped_threshold.shape

    # Create a 300x300 all black background image, and paste the 200x200 drawing in the center of the background image.
    background_image = np.zeros((300, 300), dtype="uint8")
    offset_x = round((300 - w) / 2)
    offset_y = round((300 - h) / 2)
    background_image[offset_y:offset_y + h, offset_x:offset_x + w] = cropped_threshold 

    cv2.imwrite('drawn_image.png', background_image)

def guess_number():
    neural_net = Net()
    neural_net.load_state_dict(torch.load('model.pth'))

    save_drawing_canvas()
    image = cv2.imread('drawn_image.png', cv2.IMREAD_GRAYSCALE)
    image = image[newaxis, :, :]
    image = torch.tensor(image).float()

    output = neural_net(image)
    print(output)

    probability = nnf.softmax(output, dim = 1)
    print(probability)

    top_probabilities, top_labels = torch.topk(probability, 10)
    print(top_probabilities)
    print(top_labels)
    
    _, predicted = torch.max(output.data, 1) # the class with the highest energy is what we choose as prediction
    result_label.config(text=f'This number is a {predicted.numpy()[0]}')

root = Tk()
outer_canvas = Canvas(root, width=500, height=500)
outer_canvas.pack()

drawing_canvas = Canvas(outer_canvas, width=300, height=300)
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
