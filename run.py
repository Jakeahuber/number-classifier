import PIL.Image
from neural_net import Net
import torch
import cv2
import numpy as np
import torch.nn.functional as nnf
from tkinter import * 
import imutils
from numpy import newaxis
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt

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
    outer_canvas.create_window(250, 258, window=drawing_canvas)

def save_drawing_canvas():
    try:
        drawing_canvas.postscript(file='drawn_image.ps')
        ps_image = PIL.Image.open('drawn_image.ps')
        ps_image.save('drawn_image.png')
        center_image()
    except:
        result_label.config(text="   Draw a number") # Spacing is added to make the text appear centered

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
    probability = nnf.softmax(output, dim = 1)

    probability = probability.detach().numpy()
    probability = probability[0].tolist()
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plot_chart(x, probability)
    
    _, predicted = torch.max(output.data, 1) # the class with the highest energy is what we choose as prediction
    result_label.config(text=f'This number is a {predicted.numpy()[0]}')

def plot_chart(x, y):
    fig.clear()
    plot = fig.add_subplot(111)
    plot.bar(x, y)
    plot.set_xticks(np.arange(0, 10, step=1.0))
    plot.set_yticks(np.arange(0, 1, step=0.1))
    plot.set_xlabel('Number')
    plot.set_ylabel('Probability')
    plot.set_facecolor('#F0F0F0')
    plotting_canvas.draw_idle()

root = Tk()
outer_canvas = Canvas(root, width=400, height=525)
outer_canvas.pack()

drawing_canvas = Canvas(outer_canvas, width=300, height=300)
drawing_canvas.place(x=100, y=200)
create_drawing_canvas()

# Add buttons and text to the outer canvas
btn1 = Button(outer_canvas, text='Guess Number', command=guess_number)
btn1.configure(font = ("Arial", 12), padx=3, pady=3)
btn1.place(x=120, y=425)

btn2 = Button(outer_canvas, text='Clear Drawing', command=clear_drawing_canvas)
btn2.configure(font = ("Arial", 12), padx=3, pady=3)
btn2.place(x=260, y=425)

label = Label(root, text="Number classifier")
label.configure(font = ("Arial", 20))
label.place(x=345,y=33)  

result_label = Label(outer_canvas, text="", font = ("Arial", 13))
result_label.place(x=181, y=75)


fig = Figure(figsize=(4, 4), dpi=95)
fig.patch.set_facecolor('#F0F0F0')
plot = fig.add_subplot(111)
plot.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
plot.set_xticks(np.arange(0, 10, step=1.0))
plot.set_yticks(np.arange(0, 1, step=0.1))
plot.set_xlabel('Number')
plot.set_ylabel('Probability')
plot.set_facecolor('#F0F0F0')

plotting_canvas = FigureCanvasTkAgg(fig, master=root)
plotting_canvas.draw()

plotting_canvas.get_tk_widget().pack(side='left', padx=50, pady=(0, 20))
outer_canvas.pack(side='left')

root.mainloop()
