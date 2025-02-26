import cv2
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq

class BackgroundRemover:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    def remove_background(self, image):
        """
        Removes background while preserving color using MOG2.
        """
        fg_mask = self.bg_subtractor.apply(image)
        fg_mask = cv2.medianBlur(fg_mask, 5)  
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)  
        foreground = cv2.bitwise_and(image, image, mask=fg_mask)
        return foreground
    
    
class ColorMatcher:
    """
    Class for performing color-based image matching using HSV histograms.
    It computes both global and local color histograms for robust matching.
    """
    def __init__(self, h_bins=30, s_bins=32, v_bins=16, grid_size=(4, 4)):
        self.h_bins = h_bins  # Hue bins
        self.s_bins = s_bins  # Saturation bins
        self.v_bins = v_bins  # Value bins
        self.grid_size = grid_size

    def _compute_histogram(self, hsv, bins, normalize=True):
        hist = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
        if normalize:
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def compute_features(self, image):
        """
        Compute global and local color histograms as feature vectors.
        """
        image = cv2.resize(image, (512, 512))  # Resize for consistency
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
        features = []

        # Global histogram
        global_hist = self._compute_histogram(hsv, [self.h_bins, self.s_bins])
        features.append(global_hist.flatten())

        # Local histograms
        h, w = hsv.shape[:2]
        grid_h, grid_w = h // self.grid_size[0], w // self.grid_size[1]
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                cell = hsv[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                local_hist = self._compute_histogram(cell, [self.h_bins, self.s_bins])
                features.append(local_hist.flatten())

        return np.concatenate(features)  # Concatenate all histograms into a single feature vector

    def compute_distance(self, features1, features2):
        """
        Compute a weighted distance between two feature vectors using global and local histograms.
        """
        feature_length = self.h_bins * self.s_bins
        global_f1, global_f2 = features1[:feature_length], features2[:feature_length]
        local_f1, local_f2 = features1[feature_length:], features2[feature_length:]

        global_dist = 1 - cv2.compareHist(global_f1.astype(np.float32), global_f2.astype(np.float32), cv2.HISTCMP_CORREL)
        local_dist = cv2.compareHist(local_f1.astype(np.float32), local_f2.astype(np.float32), cv2.HISTCMP_CHISQR_ALT)

        return 0.4 * global_dist + 0.6 * local_dist

    def find_matches(self, query_features, dataset_features, top_n=5):
        """
        Find the top-N best matches from the dataset based on color similarity.
        """
        distances = [(filename, self.compute_distance(query_features, features)) for filename, features in dataset_features.items()]
        return heapq.nsmallest(top_n, distances, key=lambda x: x[1])  # Efficient sorting


class OrbMatcher:
    """
    Class for generating ORB keypoints and descriptors and finding image matches based 
    on ORB feature comparison using a brute-force matcher.
    It matches a query image to a dataset by counting keypoint matches.
    """
    def __init__(self):
        """
        Initializes the OrbMatcher class with the ORB feature detector.
        """
        self.orb = cv2.ORB_create()  # Initialize ORB detector

    def generate_orb_descriptors(self, image):
        """
        Compute ORB keypoints and descriptors for an image (NumPy array).
        """
        if isinstance(image, str):  # If a file path is provided, load the image
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Error: Could not load image {image}.")
        else:
            img = image  # If already an image, use it directly

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        descriptors = descriptors if descriptors is not None else np.array([], dtype=np.uint8)

        return keypoints, descriptors


    def save_orb_descriptors(self, image_path, output_filename):
        """
        Compute ORB descriptors and save them to a file.
        """
        keypoints, descriptors = self.generate_orb_descriptors(image_path)
        with open(output_filename, 'wb') as f:
            pickle.dump((keypoints, descriptors), f)  # Save serializable keypoints and descriptors

    def load_descriptors(self, descriptor_file):
        """
        Load ORB descriptors from a file.
        """
        with open(descriptor_file, 'rb') as f:
            keypoints, descriptors = pickle.load(f)
        return keypoints, descriptors

    def get_orb_matches(self, query_image_path, descriptor_dir):
        """
        Find images in a dataset with similar ORB features to a query image.
        """
        # Generate descriptors for the query image
        query_keypoints, query_descriptors = self.generate_orb_descriptors(query_image_path)
        
        image_matches = {}

        # Iterate through the dataset descriptors and match ORB features
        for filename in os.listdir(descriptor_dir):
            if filename.endswith('.pkl'):
                existing_desc_file = os.path.join(descriptor_dir, filename)
                _, existing_descriptors = self.load_descriptors(existing_desc_file)

                if query_descriptors is None or existing_descriptors is None or len(query_descriptors) == 0 or len(existing_descriptors) == 0:
                    continue

                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(query_descriptors, existing_descriptors)

                # Store the number of matches for this image
                image_matches[filename] = len(matches)

        return image_matches


def hybrid_matching(dataset_path, query_image_path, background_path, descriptor_path, top_n_orb=15, top_n_color=6):
    bg_remover = BackgroundRemover()
    orb_matcher = OrbMatcher()
    color_matcher = ColorMatcher()

    # Load & process query image
    query_image = cv2.imread(query_image_path)
    query_cleaned = bg_remover.remove_background(query_image)
    query_features = color_matcher.compute_features(query_cleaned)
    query_descriptors = orb_matcher.generate_orb_descriptors(query_cleaned)


    # Compute ORB features
    orb_matches = orb_matcher.get_orb_matches(query_image_path, descriptor_path)
    top_orb_matches = heapq.nlargest(top_n_orb, orb_matches.items(), key=lambda x: x[1])


    dataset_features = defaultdict(list)
    dataset_images = defaultdict(list)
    orb_match_counts = defaultdict(int)
    results = []

    for filename, _ in top_orb_matches:
        image_name = filename.replace('.pkl', '.jpg')
        image_path = os.path.join(background_path, image_name)  # Use background-removed version

        image = cv2.imread(image_path)
        if image is None:
            continue

        dataset_images[filename] = cv2.imread(os.path.join(dataset_path, image_name))  # Store original for display
        dataset_features[filename] = color_matcher.compute_features(image)

        _, existing_descriptors = orb_matcher.load_descriptors(os.path.join(descriptor_path, filename))
        query_keypoints, query_descriptors = orb_matcher.generate_orb_descriptors(query_image_path)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(query_descriptors, existing_descriptors)
        orb_match_counts[filename] = len(matches)

    color_matches = color_matcher.find_matches(query_features, dataset_features, top_n=top_n_color)

    max_color_distance = 255  
    max_orb_matches = len(query_features)  

    weight_color = 0.5  
    weight_orb = 0.5  

    for filename, distance in color_matches:
        orb_match_count = orb_match_counts[filename]

        normalized_color_distance = min(distance / max_color_distance, 1)  
        normalized_orb_matches = orb_match_count / max_orb_matches if max_orb_matches else 0

        combined_score = weight_color * (1 - normalized_color_distance) + weight_orb * normalized_orb_matches
        results.append((filename, distance, orb_match_count, combined_score))

    results.sort(key=lambda x: x[3], reverse=True)

    num_matches = len(results)
    plt.figure(figsize=(13, 4))  

    plt.subplot(3, num_matches + 1, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(query_image_path), cv2.COLOR_BGR2RGB))  
    plt.title('Query Image')
    plt.axis('off')

    for i, (filename, distance, orb_match_count, combined_score) in enumerate(results, 1):
        plt.subplot(3, num_matches + 1, i + 1)
        plt.imshow(cv2.cvtColor(dataset_images[filename], cv2.COLOR_BGR2RGB))  
        plt.title(f'Match {i}')
        plt.axis('off')

        metrics_text = f'Color Distance: {distance:.3f}\nORB Matches: {orb_match_count}\n'
        combined_score_text = f'Combined Score: {combined_score:.3f}'

        plt.subplot(3, num_matches + 1, num_matches + i + 2)
        plt.text(0.5, 0.65, metrics_text, ha='center', va='center', fontsize=10)  
        plt.text(0.5, 0.35, combined_score_text, ha='center', va='center', fontsize=9, fontweight='bold')  
        plt.axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(description='Hybrid ORB and Color Matching')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--query', type=str, required=True, help='Path to the query image')
    parser.add_argument('--background', type=str, required=True, help='Directory with the images without background.')
    parser.add_argument('--descriptor', type=str, required=True, help='Directory with ORB descriptor files')
    parser.add_argument('--top_orb', type=int, default=10, help='Number of top ORB matches to use (default: 15)')
    parser.add_argument('--top_color', type=int, default=5, help='Number of top color matches to use (default: 6)')

    args = parser.parse_args()

    hybrid_matching(args.dataset, args.query, args.background, args.descriptor, top_n_orb=args.top_orb, top_n_color=args.top_color)
