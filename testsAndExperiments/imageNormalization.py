import cv2
import os
import numpy as np
import argparse

def standardize_image(image):
    """
    Standardizes the image by:
    - Resizing to a standard size.
    - Applying CLAHE (only to the L channel in LAB color space) for contrast enhancement.
    - Optionally reducing noise with GaussianBlur.
    """
    image = cv2.resize(image, (512, 512))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    
    standardized_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    standardized_image = cv2.GaussianBlur(standardized_image, (5, 5), 0)

    return standardized_image

def remove_background_from_image(image_path, background_subtractor):
    """
    Removes the background while keeping the foreground (car) in full color.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}.")
        return None

    standardized_image = standardize_image(image)
    fg_mask = background_subtractor.apply(standardized_image)

    fg_mask = cv2.medianBlur(fg_mask, 5)  
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)  # Enhance separation

    foreground = cv2.bitwise_and(standardized_image, standardized_image, mask=fg_mask)

    return foreground

def process_dataset(input_folder, output_folder):
    """
    Processes all images in the dataset, removes the background while keeping color,
    and saves the cleaned images to an output folder.
    """
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            foreground_image = remove_background_from_image(input_image_path, background_subtractor)

            if foreground_image is not None:
                cv2.imwrite(output_image_path, foreground_image)
                print(f"Processed and saved: {filename}")
            else:
                print(f"Skipping image due to error: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset images normalization')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset with images')
    parser.add_argument('--output', type=str, required=True, help='Path where foreground images will be saved')
    
    args = parser.parse_args()
    
    process_dataset(args.dataset, args.output)
