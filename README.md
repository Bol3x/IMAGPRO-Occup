# Occupation Detection System

This program detects space occupancy by using image processing techniques (background subtraction and filtering) to determine whether a given location is occupied or not.

This requires prior information of regions of interest via bounding boxes, and assumes that the video is static throughout the process.

The project uses the following libraries:
- Numpy
- OpenCV
- Seaborn
- matplotlib
- scikit-learn

## Demo
To demo the project, run `main.py` and let the program run until the end of the video

To use your own videos, you need to annotate Box locations for each seat and import it into the program as an XML file.

Below are sample images of the program execution:

![image](https://github.com/Bol3x/IMAGPRO-Occup/assets/59347516/227d2dfb-6c1a-4d94-9db4-3a4f4810b0cd)

![image](https://github.com/Bol3x/IMAGPRO-Occup/assets/59347516/915482c7-4a9f-4672-bb00-89b95baa78df)

Below are the confusion matrix results of the demo execution:

![conf_matrix_roi_overall](https://github.com/Bol3x/IMAGPRO-Occup/assets/59347516/13fde379-f084-41b0-9fde-489c8f396814)

The system has an accuracy of 65.6% using our annotated ground truths as reference.
