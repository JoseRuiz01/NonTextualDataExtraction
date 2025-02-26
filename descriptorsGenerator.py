import cv2
import numpy as np
import pickle
import os
import argparse


def keypoints_to_list(keypoints):
    """
    Convert OpenCV keypoint objects to serializable format.

    This function transforms a list of OpenCV keypoint objects into a list of tuples
    containing the essential properties of each keypoint. This conversion is necessary
    for serialization since OpenCV keypoint objects cannot be directly pickled.
    """
    return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]


def save_descriptors(filename, keypoints, descriptors):
    """
    Serialize and save keypoints and descriptors to a pickle file.

    This function first converts keypoints to a serializable format using the keypoints_to_list
    function, then saves both the converted keypoints and their corresponding descriptors
    to a pickle file for later use in feature matching or image recognition tasks.
    """
    kp_list = keypoints_to_list(keypoints)
    with open(filename, 'wb') as f:
        pickle.dump((kp_list, descriptors), f)


def generate_orb_descriptors(image_path, output_filename):
    """
    Generate ORB (Oriented FAST and Rotated BRIEF) descriptors for an image.

    This function loads an image, converts it to grayscale, and uses OpenCV's ORB
    algorithm to detect keypoints and compute their descriptors. The results are
    then saved to a pickle file. ORB is a fast, rotation-invariant feature detector
    that can be used for object recognition, image matching, and tracking.

    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}.")
        return

    # Convert to grayscale as ORB works on grayscale images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector with default parameters
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Handle the case when no descriptors are found
    if descriptors is None or len(descriptors) == 0:
        descriptors = np.array([], dtype=np.uint8)
    else:
        descriptors = descriptors.astype(np.uint8)

    # Save the results to the specified output file
    save_descriptors(output_filename, keypoints, descriptors)
    print(
        f"Descriptors saved to {output_filename}: keypoints: {len(keypoints)} - descriptors: {len(descriptors)}")


def process_directory(directory, output):
    """
    Process all JPG images in a directory and generate ORB descriptors for each.

    This function iterates through all JPG files in the specified directory,
    generates ORB descriptors for each image, and saves the results to the
    output directory with the same filename but a .pkl extension.
    """
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg'):
            # Construct full paths for input and output files
            image_path = os.path.join(directory, filename)
            output_filename = os.path.join(
                output, f"{os.path.splitext(filename)[0]}.pkl")

            # Generate and save descriptors for the current image
            generate_orb_descriptors(image_path, output_filename)


if __name__ == "__main__":
    """
    Main entry point for the script when executed directly.

    Sets up command-line argument parsing for input dataset directory and
    output directory, then processes all images in the dataset directory
    to generate and save ORB descriptors.
    """
    parser = argparse.ArgumentParser(description='Descriptors Generator')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Directory containing the image dataset (.jpg files)')
    parser.add_argument('--output', type=str, required=True,
                        help='Directory where descriptor files (.pkl) will be saved')
    args = parser.parse_args()

    process_directory(args.dataset, args.output)
