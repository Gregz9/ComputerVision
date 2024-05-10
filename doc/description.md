# Project Description: 3D reconstruction and modelling of environment using VSLAM 

### Members of te group: 
I will be the only member of the project group.
Name: Grzegorz D. Kajda
UiO-username: grzegork

### Project description
During my master's degree, I will be writing a thesis about the application of deep learning techniques to agricultural robotics. The specific aim of the thesis will be to create an algorithm which utilises egocentric vision to learn how to operate its arm(-s) to perform various tasks including picking crops, spraying pesticide if necessary, and lastly to navigate in the field, i.e. the environment. In order to achieve this, the deep-learning model will need to be trained and tested in a simulation, which again will require a model of the environment. Hence, my proposal for the project is the usage of VSLAM to reconstruct an arbitrary environment, e.g. a room, in 3D, to form a basis for creating a simulation in Unity or Unreal Engine using the VSLAM model. Once a 3d reconstruction is achieved, I aim at using a NeRF neural network, such as InstantNGP, to create another 3D reconstruction given some of the data used in VLSAM and compare the achieved results. While building a deep neural network capable of creating neural radiance fields, i.e. 3D reconstructions given 2D images would be very interesting, training such a model is extremely challenging. The amount of data needed to be collected, time required to annotate and preprocessing, let alone the diversity of data required would require far more time than one month for the project.

### Equipment needed
I have an Iphone camera which I can utilize for the purpose of VSLAM, although if possible, it definetely would be preferable to have access to a stereo camera which would allow for better depth information. Additionally, while not mentioned as part of the project, it would be very interesting to have access to a simple LiDAR sensor to perform standard SLAM against VSLAM. I should have all the other resources, such as access to GPUs at UiO which will be beneficial when running computations.

 
