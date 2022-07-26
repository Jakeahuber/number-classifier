from tkinter import * 
import PIL.Image
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import h5py

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

def save():
    try:
        label = int(label_input.get("1.0","end-1c"))
    except:
        print("enter number for label.")
        return
    accepted_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if label not in accepted_values:
        print("enter a number 0-9")
        return
    save_drawing_canvas() # Saves canvas to drawn_image.png
    
    image = cv2.imread('drawn_image.png', cv2.IMREAD_GRAYSCALE)
    with h5py.File('trainset.hdf5', 'a') as f:
        print(len(f.keys()))
        g = f.create_group(str(len(f.keys())))
        g.create_dataset('image', data = image)
        g.create_dataset('label', data = np.array([label]))


root = Tk()
outer_canvas = Canvas(root, width=500, height=500)
outer_canvas.pack()

drawing_canvas = Canvas(outer_canvas, width=300, height=300)
create_drawing_canvas()

# Add buttons and text to the outer canvas
btn = Button(root, text='Save', command=save)
btn.place(x=225, y=435)

btn = Button(root, text='Clear Drawing', command=clear_drawing_canvas)
btn.place(x=200, y=460)

label_input = Text(root, height = 1, width = 31)
label_input.place(x=140, y=410)
label_for_label_input = Label(root, text = "Label:")
label_for_label_input.place(x=100, y=410)

label = Label(text="Number classifier")
label.place(x=200,y=50)  

result_label = Label(text="")
result_label.place(x=200, y=75)

root.mainloop()