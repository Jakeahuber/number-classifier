# Number Classifier
PyTorch program for classifying handwritten digits 0-9. 

![demonstration](https://user-images.githubusercontent.com/68114979/181060279-a6b4ff6a-3fab-4b5d-bd01-588332e99301.gif)

---
- This project trains a convolutional neural network to classify handwritten digits 0-9. Users can see the model in action when executing run.py. This file displays a GUI that allows users to draw their own numbers. When the user finishes drawing the number, the GUI displays a bar chart of the neural network’s predicted output probabilities (as seen above).

- My goal for this project was to get comfortable with the PyTorch framework, which I used to build, train, and test the machine learning model. I am using this framework for my undergraduate research, but I never used PyTorch before starting this project

- I collected the data for this project’s training and testing set by executing add_data_training_set.py. This file displays a GUI that allows a user to draw a number and add its corresponding label. After the user finishes drawing, the program saves the image and label to either dataset.hdf5 (the training set), or testingset.hdf5. This project is still in the process of collecting data. 

- The project saves the images as 300x300 pixel grayscale images. I computed a bounding box for the unedited drawn digit from the user and created a new cropped image containing only the contents of the bounding box. I then rescaled this new image to fit within a 200x200 pixel box and generated the final output by pasting this rescaled image into the center of a 300x300 pixel all-black image.    

- This project taught me the importance of collecting data for a machine learning model in a consistent manner throughout a project. This project initially used training and testing sets from "The MNIST Database of Handwritten Digits" by Yann LeCun, Corinna Cortes, and Christopher H.C. Burges. My model performed well on the MNIST testing set, with about 91% accuracy. However, the model's performance was poor when using the digits I had drawn in the run.py GUI. I realized there was likely a significant difference between the method I used to save the handwritten digits in run.py and the method the MNIST database used. Subsequently, I decided to create my own training and testing sets and saw the performance of my model with handwritten digits in run.py increase dramatically. 

- I used [PyTorch's deep learning tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py) as a framework for building my convolutional neural network. 
