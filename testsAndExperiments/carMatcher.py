import cv2
import numpy as np
import pickle
import os

import sys


pathInput = sys.argv[1]
pathOutput = sys.argv[2]
mode = sys.argv[3]

#
#def orb_feature_matching(img1_path, img2_path):
#    # Load images
#    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
#    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
#    
#    # Check if images are loaded
#    if img1 is None or img2 is None:
#        print("Error: Could not load images.")
#        return
#    
#    # Initialize ORB detector
#    orb = cv2.ORB_create()
#    
#    # Find keypoints and descriptors
#    kp1, des1 = orb.detectAndCompute(img1, None)
#    kp2, des2 = orb.detectAndCompute(img2, None)
#    
#    # Create BFMatcher (Brute-Force Matcher)
#    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#    
#    # Match descriptors
#    matches = bf.match(des1, des2)
#    
#    # Sort matches by distance
#    matches = sorted(matches, key=lambda x: x.distance)
#    
#    # Draw top 20 matches
#    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#    
#    # Display result
#    cv2.imshow("ORB Feature Matching", img_matches)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#
## Example usage
#orb_feature_matching('image1.jpg', 'image2.jpg')

dataDirectory= './outputDescriptors'

#def save_descriptors(filename, keypoints, descriptors):
#    with open(filename, 'wb') as f:
#        pickle.dump((keypoints, descriptors), f)
#
#def load_descriptors(filename):
#    with open(filename, 'rb') as f:
#        return pickle.load(f)

def keypoints_to_list(keypoints):
    return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

#def list_to_keypoints(kp_list):
#    return [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], _size=pt[1], _angle=pt[2], _response=pt[3], _octave=pt[4], _class_id=pt[5]) for pt in kp_list]

def list_to_keypoints(kp_list):
    return [cv2.KeyPoint(pt[0][0], pt[0][1], pt[1], pt[2], pt[3], int(pt[4]), int(pt[5])) for pt in kp_list]


def save_descriptors(filename, keypoints, descriptors):
    kp_list = keypoints_to_list(keypoints)
    with open(filename, 'wb') as f:
        pickle.dump((kp_list, descriptors), f)

def load_descriptors(filename):
    with open(filename, 'rb') as f:
        kp_list, descriptors = pickle.load(f)
    return list_to_keypoints(kp_list), descriptors

def generate_orb_descriptors(image_path, output_filename):
    # Load image in color
    img = cv2.imread(image_path)
    
    # Check if image is loaded
    if img is None:
        print(f"Error: Could not load image {image_path}.")
        return
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Find keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)


    # Ensure descriptors are of type np.uint8
    if  descriptors is None or len(descriptors) == 0:
        descriptors = np.array([], dtype=np.uint8)
    else: 
        descriptors = descriptors.astype(np.uint8)
        


    # Save descriptors
    save_descriptors(output_filename, keypoints, descriptors)
    print(f"Descriptors saved to {output_filename}: keypoints: {len(keypoints)} - descriptors: {len(descriptors)}")

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            output_filename = os.path.join(dataDirectory, f"{os.path.splitext(filename)[0]}.pkl")
            generate_orb_descriptors(image_path, output_filename)

def compare_orb_descriptors(file1, file2):
    # Load descriptors from files
    kp1, des1 = load_descriptors(file1)
    kp2, des2 = load_descriptors(file2)

    # Create BFMatcher (Brute-Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    print(f"Found {len(matches)} matches.")
    return matches


def compare_new_image_with_directory(new_image_path):
    # Generate ORB descriptors for the new image
    new_image_desc_file = "new_image.pkl"
    generate_orb_descriptors(new_image_path, new_image_desc_file)
    
    # Load new image descriptors
    _, new_des = load_descriptors(new_image_desc_file)
    
    best_match = None
    best_match_count = 0
    best_match_file = None
    
    for filename in os.listdir(dataDirectory):
        if filename.endswith('.pkl'):
            existing_desc_file = os.path.join(dataDirectory, filename)
            _, existing_des = load_descriptors(existing_desc_file)
            print(existing_des.shape)
            if new_des is None or existing_des is None or new_des.shape[0] == 0 or  existing_des.shape[0] == 0:
                print(f"Skipping comparison due to descriptor mismatch: new vs {filename}")
                continue 
            
            # Create BFMatcher and match descriptors
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(new_des, existing_des)
            
            # Count matches
            match_count = len(matches)
            print(f"Compared with {filename}, found {match_count} matches.")
            
            if match_count > best_match_count:
                best_match_count = match_count
                best_match = matches
                best_match_file = filename
    
    print(f"Best match: {best_match_file} with {best_match_count} matches.")
    return best_match_file, best_match


# Example usage

if(mode == "directory"):
    process_directory(pathInput)
else:
    best_match_file, matches = compare_new_image_with_directory(pathInput)



