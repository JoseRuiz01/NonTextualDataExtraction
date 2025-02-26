import cv2
import numpy as np
import pickle
import os
import argparse


def keypoints_to_list(keypoints):
    return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

def save_descriptors(filename, keypoints, descriptors):
    kp_list = keypoints_to_list(keypoints)
    with open(filename, 'wb') as f:
        pickle.dump((kp_list, descriptors), f)

def generate_orb_descriptors(image_path, output_filename):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Check if descriptors are valid
    if descriptors is None or len(descriptors) == 0:
        descriptors = np.array([], dtype=np.float32)  # Use np.float32
    else: 
        descriptors = descriptors.astype(np.float32)  # Ensure descriptors are in float32 format
    
    save_descriptors(output_filename, keypoints, descriptors)
    print(f"Descriptors saved to {output_filename}: keypoints: {len(keypoints)} - descriptors: {len(descriptors)}")

def process_directory(directory, output):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            output_filename = os.path.join(output, f"{os.path.splitext(filename)[0]}.pkl")
            generate_orb_descriptors(image_path, output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Descriptors Generator')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    process_directory(args.dataset, args.output)
