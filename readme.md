# Guide to Using the CBIR System

To use the CBIR system for finding similar car's images to your car photo, follow these simplified steps:

#### Step 1: Install Required Software

1. **Install Python**: Make sure Python is installed on your computer. You can download it from https://www.python.org/.

2. **Install Necessary Libraries**: Open your terminal or command prompt and run:

   pip install opencv-python numpy matplotlib

#### Step 2: Prepare Your Image Dataset

1. **Organize Your Images**: Create a directory and place all your JPG images in this directory. You can use the Standford cars dataset : https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset

#### Step 3: Generate ORB Descriptors

1. **Open Terminal or Command Prompt**: Navigate to the directory containing the `descriptorsGenerator.py` script.

2. **Run the Descriptors Generator**: Execute the following command to generate ORB descriptors:

   python descriptorsGenerator.py --dataset path_to_image_dataset --output path_to_output_directory

   - Replace `path_to_image_dataset` with the path to your image directory.
   - Replace `path_to_output_directory` with the path where you want to save the descriptor files.

#### Step 4: Generate the Output Image with Matches

1. **Run the CBIR Script**: Use the `CBIR.py` script to generate an output image with matches. You will need to specify the query image and the dataset directory.

2. **Command to Run**:

   python CBIR.py --query path_to_query_image --dataset path_to_image_dataset

   - Replace `path_to_query_image` with the path to your query image.
   - Replace `path_to_image_dataset` with the path to your image dataset directory.

By following these steps, you can generate an output image that shows the matches for your query image using the CBIR system.
