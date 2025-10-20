# Efficient-Optical-Neural-Network-Training
This project uses a optics-informed backpropagation algorithm to train diffractive neural network architectures efficiently. The following features are available in the ONNSupplemental.ipynb file:

1) Generation of MNIST training data sets with adjustable image/optical set up parameters
2) Adjustable optical neural network architecture
3) Optical neural network training algorithm
4) Arbitrary optical linear transformation generator

If you use this code, we ask that you cite this preprint:

Bart, Manon P., Nick Sparks, and Ryan T. Glasser. "Efficient Training for Optical Computing." arXiv preprint arXiv:2506.20833 (2025).

## Table of Contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Upcoming Changes](#upcoming-changes)

## Introduction
This project simulates and trains optical neural networks with cascaded phase masks. This work is based on our project which leverages the Fourier Transform for efficient training. 

**Optical Architecture**

The ONN architecture is shown in the following figure (d):

<img width="484" height="310" alt="Screenshot 2025-10-10 at 6 40 40 PM" src="https://github.com/user-attachments/assets/683ad1f7-258d-442a-8f98-a5df2e9731b6" />

where the blue corresponds to the phase/amplitude of light and the red corresponds to phase masks (for SLMs or other devices capable of modulating phase). The goal is to determine the optimal phase masks for optical machine learning inference tasks.

The linear transformations at each layer follow the angular spectrum method: 

<img width="297" height="96" alt="Screenshot 2025-10-10 at 6 40 07 PM" src="https://github.com/user-attachments/assets/d13b290b-5085-4677-b9cf-b1bbdeb5ac9d" />


**The Training Algorithm**

The training algorithm uses a Fourier-decomposition based backpropagation algorithm. The calculations for the gradient are as follows:

<img width="382" height="185" alt="Screenshot 2025-10-10 at 6 39 05 PM" src="https://github.com/user-attachments/assets/b601511f-d424-4091-b6be-f929ab185855" />

Beyond the novel training algorithm, several features are available in the code to stabilize learning for optical architectures including gradient normalization, learning rate decay, auxillary gradient stabilization functions, and categorical cross entropy loss. Further information on the physics behind the algorithm, wokring architecture examples and features is available in the aformentioned preprint.

---

In summary, this repository lets you:
1. Create ONN architecures and training data sets
2. Generate phase masks for ONN inference tasks
3. Generate phase masks for arbitrary linear transformations between two field-of-views
4. Supplies visualization tools for data and results


## Usage

Examples of the capabilities are available in the ONNSupplemental.ipynb file.

### Generation of data:
The class CreateImages has several helpful features to create ONN data sets. The number of training images (n_imag), number of testing images (n_img_test), size of the image (size_image), digits used (digits), amount of padding (padding) and scale of the detector regions (detector_scale).

          C = CreateImages(n_imag, n_img_test, size_image, digits=[0,1,2,3,4,5,6,7,8,9],padding=8,detector_scale=1)

The size_image will create a square image. The digits call will filter only the digits you want to work with and automatically adjust the number of detector regions. This could alternatively use the Fashion MNIST dataset. The default size of the detector regions corresponds to detector_scale = 1, however it can be adjusted (i.e. .8, 1.2, etc) to change the size.  

The images are then created using:

          x, y, y_label, x_test, y_test, y_l_test  = C.process()
  
where x is the training images, y is the target output image, and y_label is the integer valued label. The resulting data sets can be visualized using:

          C.plotimage(x,y,y_label)

<img width="630" height="337" alt="1f069a16-e3d3-409f-850b-4f77f959ce6d" src="https://github.com/user-attachments/assets/c2e1c7c4-b9ab-47a0-9b9f-79e4d39693fa" />

The grid locations can be visualized using:

          C.visualize_grids()
  
<img width="508" height="454" alt="514b79a7-ac7d-4dc9-b833-b5f546f2d614" src="https://github.com/user-attachments/assets/b27180ad-6354-49b7-a60f-b1fc81e78e96" />

Their size is estimated using a default pixel size and field-of-view length (ps = 8e-6, length = 1000). The pixel_size is automatically adjusted when calling size_image. This can be changed manually inside the code. Finally, 

          defined_grids = C.return_grids()

is called in order to save the grid locations of each detector region. These grid locations will be used in training. 

### Optical Neural Network Training:

The classifying ONN can be created using:

          ONNd = ONN_Class_Det(size, n_L, sub_grids, wavelength, z, pixel_size) 

where the size of the image, the number of phase masks (layers of your ONN), sub grid locations, wavelength, distance between layers, and pixel size are encoded. 

This generates the optical architecture. To train the architecture, you can call:

        lossd, accd, wd, gd = ONNd.train(x, y_label, epochs, batch_size, learning_rate)

which saves the loss and accuracy over epochs, as well as the final phase masks  (wd) and gradient (gd). To encode the input information in the phase instead of amplitude you may optionally call:

        lossd, accd, wd, gd = ONNd.train(np.exp(1j*x*2*np.pi), y_label, epochs, batch_size, learning_rate)

The final optimized phase masks are available in wd. 

The accuracy and loss over epochs can be visualized using:

        ONNd.plot_la(lossd, accd)

<img width="790" height="590" alt="4b987fae-dfdf-4056-9bb3-6731e02667a2" src="https://github.com/user-attachments/assets/3a9217c5-c87a-4469-8b09-e85a9f09a8a3" />

The confusion matrix can be visualized using:

        ONNd.plot_cm(x_test,y_l_test)
        
<img width="644" height="543" alt="ce3f192d-1da3-4e47-a9df-708b5eac6423" src="https://github.com/user-attachments/assets/2cc24455-0d4a-4385-94a6-253aa44163bb" />

The final output can be visualized using:

        ONNd.plot_final_output(x_test,y_test,y_l_test,162)
        
<img width="590" height="219" alt="424e3e19-d1aa-4e53-9289-ddd4462a5818" src="https://github.com/user-attachments/assets/38b8f4ee-cff4-449f-80fd-ba5cd6f8fde0" />
<img width="590" height="219" alt="7f89af52-8371-4087-8a59-d6c0ada35de5" src="https://github.com/user-attachments/assets/f8713ce1-b286-4849-8674-32f898b41c30" />


Finally, the detector region can be visualized using:

        ONNd.plot_detector_distribution(x_test,y_l_test,40)
        
<img width="590" height="240" alt="30ced111-7cc3-40d0-8be1-3cd78fa45fe0" src="https://github.com/user-attachments/assets/e0c9d6d3-e5c6-447c-b4c2-e580a4343ee2" />

### Optical Linear Transformations:

Optical linear transformations between two arbitrary amplitude and/or phase encoded field of views may also be generated using the code provided, where the training will determine the phase masks necessary for the transformation. Below are examples of several linear transformations.

<img width="590" height="220" alt="84b71c61-ba52-4f76-b7fa-cf96a0d83b88" src="https://github.com/user-attachments/assets/0611d8e2-78ba-4666-a7b6-d720150831f5" />

<img width="898" height="202" alt="fceb2e70-385b-4d0e-8237-3573deebb36e-1" src="https://github.com/user-attachments/assets/9dcc688c-ca2c-4931-9755-6166da086673" />

An example of the input encoding and training is shown in the Generation of Arbitrary Linear Transformations section of ONNSupplemental.ipynb.

## Upcoming Changes

This code will soon be optimized using JAX to support faster training. 
