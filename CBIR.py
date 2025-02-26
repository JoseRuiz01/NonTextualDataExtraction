import cv2
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq


class ColorMatcher:
    """
    Class for performing color-based image matching using a combination of
    Conventional Colour Histogram (CCH) and Stacked Colour Histogram (SCH).

    The ColorMatcher implements a dual histogram approach for content-based image retrieval:
    1. CCH captures the global color distribution in RGB space
    2. SCH captures texture information by applying iterative mean filtering

    Attributes:
        bins (int): Number of histogram bins for quantizing color channels
        filter_size (int): Size of the mean filter kernel for SCH computation
        iterations (int): Number of iterative filtering steps for SCH
    """

    def __init__(self, bins=64, filter_size=7, iterations=5):
        """
        Initialize ColorMatcher with configurable parameters.
        """
        self.bins = bins
        self.filter_size = filter_size
        self.iterations = iterations

    def compute_cch(self, image):
        """
        Compute Conventional Colour Histogram (CCH) in RGB space.

        This function calculates a 3D histogram in RGB color space, providing
        a global representation of the image's color distribution.
        """
        # Standardize image size to ensure consistent feature dimensions
        image = cv2.resize(image, (512, 512))

        # Calculate 3D histogram across all three color channels
        hist = cv2.calcHist([image], [0, 1, 2], None, [
                            self.bins] * 3, [0, 256] * 3)

        # Normalize and flatten the histogram to create a feature vector
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def compute_sch(self, image):
        """
        Compute Stacked Colour Histogram (SCH) using mean filtering.

        This function applies iterative mean filtering (blurring) to capture
        texture information at multiple scales. The histograms from each
        iteration are stacked to form a multi-scale representation.
        """
        # Convert to grayscale since SCH focuses on texture, not color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initialize empty histogram for aggregating results
        sch_hist = np.zeros((self.bins,), dtype=np.float32)

        # Iteratively apply mean filtering and sum up histograms
        for _ in range(self.iterations):
            # Apply mean filter (blurring)
            image = cv2.blur(image, (self.filter_size, self.filter_size))

            # Calculate histogram of the blurred image
            hist = cv2.calcHist([image], [0], None, [self.bins], [0, 256])

            # Normalize and flatten the histogram
            hist = cv2.normalize(hist, hist).flatten()

            # Add to the accumulated histogram
            sch_hist += hist

        return sch_hist

    def compute_features(self, image):
        """
        Combine CCH and SCH into a single feature vector.

        This creates a comprehensive color and texture descriptor by
        concatenating the conventional color histogram with the
        stacked color histogram.
        """
        cch = self.compute_cch(image)
        sch = self.compute_sch(image)
        return np.concatenate((cch, sch))

    def compute_distance(self, features1, features2, alpha=0.5):
        """
        Compute weighted similarity between two feature vectors.

        This function calculates the distance between two images by comparing
        their CCH and SCH components separately, then combines them with a
        weighted average.
        """
        # Split the feature vectors into CCH and SCH components
        split_idx = len(features1) // 2
        cch1, sch1 = features1[:split_idx], features1[split_idx:]
        cch2, sch2 = features2[:split_idx], features2[split_idx:]

        # Calculate correlation-based similarity for both components
        cch_dist = cv2.compareHist(cch1.astype(
            np.float32), cch2.astype(np.float32), cv2.HISTCMP_CORREL)
        sch_dist = cv2.compareHist(sch1.astype(
            np.float32), sch2.astype(np.float32), cv2.HISTCMP_CORREL)

        # Combine CCH and SCH distances with weighting factor alpha
        return alpha * (1 - cch_dist) + (1 - alpha) * (1 - sch_dist)

    def find_matches(self, query_features, dataset_features, top_n=5):
        """
        Find the top-N best matches from the dataset based on color similarity.
        """
        # Calculate distances between query and all dataset images
        distances = [(filename, self.compute_distance(query_features, features))
                     for filename, features in dataset_features.items()]

        # Sort by distance (lower distance = better match) and return top N
        return sorted(distances, key=lambda x: x[1])[:top_n]


class OrbMatcher:
    """
    Class for generating ORB keypoints and descriptors and finding image matches based 
    on ORB feature comparison using a brute-force matcher.

    ORB (Oriented FAST and Rotated BRIEF) is a fast and robust local feature detector
    that identifies "keypoints" in images and computes binary descriptors for matching.

    This class handles:
    1. Extracting ORB features from images
    2. Serializing and deserializing keypoints and descriptors
    3. Finding matches between images using a brute-force matcher
    """

    def __init__(self):
        """
        Initializes the OrbMatcher class with the ORB feature detector.

        Sets up the OpenCV ORB detector and initializes tracking attributes for
        the query image's descriptors and keypoints.
        """
        # Initialize OpenCV's ORB feature detector
        self.orb = cv2.ORB_create()

        # Variables to track the number of descriptors and keypoints in the query image
        self.query_descriptors = 500  
        self.query_keypoints = 2000  

    def get_descriptors(self):
        """
        Returns the query descriptors.
        """
        return self.query_descriptors

    def generate_orb_descriptors(self, image_path):
        """
        Compute ORB keypoints and descriptors for an image and return them.
        """
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(
                f"Error: Could not load image {image_path}.")

        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        # Handle the case where no descriptors are found
        descriptors = descriptors if descriptors is not None else np.array(
            [], dtype=np.uint8)

        # Convert keypoints to serializable format (for storage in pickle files)
        keypoints_serializable = [
            (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

        return keypoints_serializable, descriptors

    def save_orb_descriptors(self, image_path, output_filename):
        """
        Compute ORB descriptors and save them to a file.
        """
        # Generate keypoints and descriptors
        keypoints, descriptors = self.generate_orb_descriptors(image_path)

        # Save to a pickle file
        with open(output_filename, 'wb') as f:
            pickle.dump((keypoints, descriptors), f)

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

        This function:
        1. Extracts ORB features from the query image
        2. Compares them to pre-computed features of dataset images
        3. Returns a dictionary of {filename: match_count} for all images
        """
        # Extract features from the query image
        self.query_keypoints, self.query_descriptors = self.generate_orb_descriptors(
            query_image_path)

        # Dictionary to store match counts for each dataset image
        image_matches = {}

        # Iterate through all descriptor files in the directory
        for filename in os.listdir(descriptor_dir):
            if filename.endswith('.pkl'):
                # Load pre-computed descriptors
                existing_desc_file = os.path.join(descriptor_dir, filename)
                _, existing_descriptors = self.load_descriptors(
                    existing_desc_file)

                # Skip if either set of descriptors is empty
                if self.query_descriptors is None or existing_descriptors is None or len(self.query_descriptors) == 0 or len(existing_descriptors) == 0:
                    continue

                # Create a brute-force matcher with Hamming distance (for binary descriptors)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

                # Match descriptors between query and dataset image
                matches = bf.match(self.query_descriptors,
                                   existing_descriptors)

                # Store number of matches for each image
                image_matches[filename] = len(matches)

        return image_matches


def hybrid_matching(dataset_path, query_image_path, descriptor_dir, top_n_orb=10, top_n_color=6):
    """
    Hybrid image matching combining ORB features and Color Histograms.

    This function implements a two-stage matching process:
    1. First filter using ORB features to identify structurally similar images
    2. Then refine with color histograms for more precise matching
    3. Compute a combined score that weighs both methods
    """
    # Initialize matchers
    orb_matcher = OrbMatcher()
    color_matcher = ColorMatcher()

    # STEP 1: ORB-based initial filtering
    # Find matches based on local features (keypoints)
    orb_matches = orb_matcher.get_orb_matches(query_image_path, descriptor_dir)

    # Keep only the top N ORB matches to reduce computation in next stage
    top_orb_matches = heapq.nlargest(
        top_n_orb, orb_matches.items(), key=lambda x: x[1])

    # STEP 2: Prepare for color-based refinement
    # Load and compute features for the query image
    query_image = cv2.imread(query_image_path)
    query_features = color_matcher.compute_features(query_image)

    # Dictionaries to store dataset information
    dataset_features = defaultdict(list)  
    dataset_images = defaultdict(list)    
    dataset_keypoints = defaultdict(list)  

    results = []

    # Process only the top ORB matches (filtered candidates)
    for filename, _ in top_orb_matches:
        # Load the image (converting from .pkl filename to .jpg)
        image_path = os.path.join(
            dataset_path, filename.replace('.pkl', '.jpg'))
        image = cv2.imread(image_path)

        # Store the image and compute its color features
        dataset_images[filename] = image
        dataset_features[filename] = color_matcher.compute_features(image)

        # Load keypoints for visualization
        keypoints, _ = orb_matcher.load_descriptors(
            os.path.join(descriptor_dir, filename))

        # Convert serialized keypoints back to OpenCV KeyPoint objects
        dataset_keypoints[filename] = [cv2.KeyPoint(pt[0], pt[1], size, angle, response, octave, class_id)
                                       for pt, size, angle, response, octave, class_id in keypoints]

    # STEP 3: Color-based matching on the filtered candidates
    color_matches = color_matcher.find_matches(
        query_features, dataset_features, top_n=top_n_color)

    # STEP 4: Combine ORB and color scores
    # Define weights for hybrid scoring
    weight_color = 0.15  
    weight_orb = 0.85    

    # Normalize ORB match count by the maximum possible
    max_orb_match_count = len(orb_matcher.get_descriptors())

    # Calculate combined scores for the top color matches
    for filename, distance in color_matches:
        orb_match_count = orb_matches.get(filename, 0)

        # Normalize ORB match count to [0,1] range
        normalized_orb_match_count = orb_match_count / max_orb_match_count

        # Compute combined score
        if weight_color == 0 or distance == 0:
            combined_score = float('inf')
        else:
            # Higher score is better:
            # - Higher ORB match count is better
            # - Lower color distance is better (hence the division)
            combined_score = (weight_orb * normalized_orb_match_count) / \
                (weight_color * distance)

        results.append((filename, distance, orb_match_count, combined_score))

    results.sort(key=lambda x: x[3], reverse=True)

    # STEP 5: Visualization
    num_matches = len(results)
    plt.figure(figsize=(13, 4))

    # Prepare the query image with keypoints for visualization
    query_keypoints, _ = orb_matcher.generate_orb_descriptors(query_image_path)
    # Convert serialized keypoints back to OpenCV KeyPoint objects
    query_keypoints = [cv2.KeyPoint(pt[0], pt[1], size, angle, response, octave, class_id)
                       for pt, size, angle, response, octave, class_id in query_keypoints]
    # Draw keypoints on the query image
    query_image_with_keypoints = cv2.drawKeypoints(
        query_image, query_keypoints, None, color=(0, 255, 0))

    # Plot the query image in the first position
    plt.subplot(3, num_matches + 1, 1)
    plt.imshow(cv2.cvtColor(query_image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('Query Image')
    plt.axis('off')

    # Plot each matching image with metrics
    for i, (filename, distance, orb_match_count, combined_score) in enumerate(results, 1):
        # Draw keypoints on the dataset image
        image_with_keypoints = cv2.drawKeypoints(
            dataset_images[filename], dataset_keypoints[filename], None, color=(0, 255, 0))

        # Image with keypoints
        plt.subplot(3, num_matches + 1, num_matches + 1 + i)
        plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.title(f'Match {i}')
        plt.axis('off')

        # Metrics and scores
        plt.subplot(3, num_matches + 1, 2 * (num_matches + 1) + i)
        metrics_text = (
            f'Color Distance: {distance:.3f}\n'
            f'ORB Matches: {orb_match_count}\n'
        )
        combined_score_text = (
            f'Combined Score: {combined_score:.3f}'
        )
        plt.text(0.5, 0.65, metrics_text, ha='center',
                 va='center', fontsize=10)
        plt.text(0.5, 0.35, combined_score_text, ha='center',
                 va='center', fontsize=9, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Parses command-line arguments and launches the hybrid matching process
    with the specified parameters.
    """
    parser = argparse.ArgumentParser(
        description='Hybrid ORB and Color Matching')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--query', type=str, required=True,
                        help='Path to the query image')
    parser.add_argument('--descriptors', type=str, required=True,
                        help='Directory with ORB descriptor files')
    parser.add_argument('--top_orb', type=int, default=10,
                        help='Number of top ORB matches to use (default: 15)')
    parser.add_argument('--top_color', type=int, default=5,
                        help='Number of top color matches to use (default: 6)')

    args = parser.parse_args()

    hybrid_matching(args.dataset, args.query, args.descriptors,
                    top_n_orb=args.top_orb, top_n_color=args.top_color)
