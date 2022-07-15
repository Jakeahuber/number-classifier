import PIL.Image
from PIL import ImageEnhance
from neural_net import Neural_Net
import torch
from torchvision import transforms
import cv2
import tkinter as tk
import numpy as np

blank_img = np.zeros((28, 28, 3), dtype='uint8')
cv2.imwrite('image.png', blank_img) # Saves image

from tkinter import * 

x = 250
y = 250
def get_coords(event):
    global x, y 
    x = event.x
    y = event.y

def draw_number(event):
    global x, y
    drawing_canvas.create_line((x, y, event.x, event.y), fill='white', width=2)
    x = event.x
    y = event.y

def guess_number():
    print("guessing...")
    save_drawing_canvas()
    batch_size = 1
    neural_net = Neural_Net()
    neural_net.load_state_dict(torch.load('model.pth'))
    image = load_image()
    outputs = neural_net(image)
    _, predicted = torch.max(outputs.data, 1) # the class with the highest energy is what we choose as prediction
    print(predicted.numpy()[0])

def save_drawing_canvas():
    drawing_canvas.postscript(file='drawn_image.ps')
    ps_image = PIL.Image.open('drawn_image.ps')
    ps_image.save('drawn_image.png')

    # Resizes image
    png_img = PIL.Image.open('drawn_image.png')
    png_img = png_img.resize((28, 28))
    enhancer = ImageEnhance.Contrast(png_img)
    png_img = enhancer.enhance(2.5)
    png_img.save('drawn_image.png')

def load_image():
    image = cv2.imread('drawn_image.png', cv2.IMREAD_GRAYSCALE)
    image = torch.Tensor(image)
    return image.view(-1, 28*28).requires_grad_()

root = Tk()
outer_canvas = Canvas(root, width=500, height=500)
outer_canvas.pack()

drawing_canvas = Canvas(outer_canvas, width=200, height=200)

# Add image to drawing canvas (child canvas of the outer canvas)
drawing_canvas.bind("<Button-1>", get_coords)
drawing_canvas.bind("<B1-Motion>", draw_number)
drawing_canvas.create_rectangle(0, 0, 400, 400, fill='black')

outer_canvas.create_window(250, 250, window=drawing_canvas)

# Add buttons and text to the outer canvas
btn = Button(root, text='Guess Number', command=guess_number)
btn.place(x=200, y=425)
label = Label(text="Number classifier")
label.place(x=200,y=50)  

root.mainloop()










'''
root = Tk()
canvas = Canvas(root, width=500, height=500)
canvas.pack()

drawing_area = Canvas(root, width=28, height=28, bg='black')
canvas.place(drawing_area, 0, 0)
drawing_area.bind("<B1-Motion>", get_coords)
drawing_area.bind("<B1-Motion>", draw_number)

img = PhotoImage(file='image.png')
canvas.create_image(250, 250, image=img)

canvas.bind("<Button-1>", get_coords)
canvas.bind("<B1-Motion>", draw_number)

btn = Button(root, text='Guess Number', command=guess_number)
btn.place(x=200, y=425)
label = Label(text="Number classifier")
label.place(x=200,y=50)  

root.mainloop()

'''
