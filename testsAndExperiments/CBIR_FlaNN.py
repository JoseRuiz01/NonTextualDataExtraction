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
    Class for performing color-based image matching using HSV histograms.
    It computes both global and local color histograms for robust matching.
    """
    def __init__(self, h_bins=30, s_bins=32, v_bins=16, grid_size=(4, 4)):
        self.h_bins = h_bins  
        self.s_bins = s_bins  
        self.v_bins = v_bins  
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
        image = cv2.resize(image, (512, 512))  
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  
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

        return np.concatenate(features)  

    def compute_distance(self, features1, features2):
        """
        Compute a weighted distance between two feature vectors using global and local histograms.
        """
        feature_length = self.h_bins * self.s_bins
        global_f1, global_f2 = features1[:feature_length], features2[:feature_length]
        local_f1, local_f2 = features1[feature_length:], features2[feature_length:]

        global_dist = cv2.compareHist(global_f1.astype(np.float32), global_f2.astype(np.float32), cv2.HISTCMP_CHISQR_ALT)
        local_dist = cv2.compareHist(local_f1.astype(np.float32), local_f2.astype(np.float32), cv2.HISTCMP_CHISQR_ALT)

        return 0.4 * global_dist + 0.6 * local_dist

    def find_matches(self, query_features, dataset_features, top_n=5):
        """
        Find the top-N best matches from the dataset based on color similarity.
        """
        distances = [(filename, self.compute_distance(query_features, features)) for filename, features in dataset_features.items()]
        return heapq.nsmallest(top_n, distances, key=lambda x: x[1]) 


class OrbMatcher:
    """
    Class for generating ORB keypoints and descriptors and finding image matches based 
    on ORB feature comparison using a FLANN matcher.
    It matches a query image to a dataset by counting keypoint matches.
    """
    def __init__(self, nfeatures=5000):
        """
        Initializes the OrbMatcher class with the ORB feature detector.
        """
        self.orb = cv2.ORB_create(nfeatures=nfeatures) 

    def generate_orb_descriptors(self, image_path):
        """
        Compute ORB keypoints and descriptors for an image and return them.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Error: Could not load image {image_path}.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        descriptors = descriptors if descriptors is not None else np.array([], dtype=np.uint8)

        keypoints_serializable = [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]
        return keypoints_serializable, descriptors

    def save_orb_descriptors(self, image_path, output_filename):
        """
        Compute ORB descriptors and save them to a file.
        """
        keypoints, descriptors = self.generate_orb_descriptors(image_path)
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
        Perform FLANN-based ORB feature matching for the query image against the dataset.
        """
        query_keypoints, query_descriptors = self.generate_orb_descriptors(query_image_path)
        image_matches = {}

        if query_descriptors is None or len(query_descriptors) < 2:
            return image_matches  
        
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH (Locality Sensitive Hashing for ORB)
                            table_number=6,  # Number of hash tables
                            key_size=12,  # Size of the hash
                            multi_probe_level=1)  # Search depth

        search_params = dict(checks=32)  
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        for filename in os.listdir(descriptor_dir):
            if filename.endswith('.pkl'):
                _, existing_descriptors = self.load_descriptors(os.path.join(descriptor_dir, filename))
                
                if existing_descriptors is None or len(existing_descriptors) < 2:
                    continue  
                
                matches = flann.knnMatch(query_descriptors, existing_descriptors, k=2)
                good_matches = [m_n[0] for m_n in matches if len(m_n) == 2 and m_n[0].distance < 0.75 * m_n[1].distance]

                image_matches[filename] = len(good_matches)

        return image_matches






def hybrid_matching(dataset_path, query_image_path, descriptor_dir, top_n_orb=15, top_n_color=6):
    """
    Hybrid image matching using ORB + Color Histograms.
    - ORB for structural feature matching (FLANN)
    - Color Histograms for fine-grained visual similarity
    - Uses a weighted score (70% ORB, 30% Color)
    """
    orb_matcher = OrbMatcher()
    color_matcher = ColorMatcher()
    
    orb_matches = orb_matcher.get_orb_matches(query_image_path, descriptor_dir)
    top_orb_matches = heapq.nlargest(top_n_orb, orb_matches.items(), key=lambda x: x[1])

    query_image = cv2.imread(query_image_path)
    query_features = color_matcher.compute_features(query_image)

    dataset_features = defaultdict(list)
    dataset_images = defaultdict(list)
    
    results = []  

    for filename, _ in top_orb_matches:
        image_path = os.path.join(dataset_path, filename.replace('.pkl', '.jpg'))
        image = cv2.imread(image_path)
        dataset_images[filename] = image
        dataset_features[filename] = color_matcher.compute_features(image)

    color_matches = color_matcher.find_matches(query_features, dataset_features, top_n=top_n_color)

    weight_color = 0.3  
    weight_orb = 0.7  

    for filename, distance in color_matches:
        orb_match_count = orb_matches.get(filename, 0)  
        combined_score = (weight_orb * orb_match_count) / (weight_color *  distance)
        results.append((filename, distance, orb_match_count, combined_score))

    results.sort(key=lambda x: x[3], reverse=True)

    num_matches = len(results)
    plt.figure(figsize=(13, 4)) 
    
    plt.subplot(3, num_matches + 1, 1)
    plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
    plt.title('Query Image')
    plt.axis('off')

    for i, (filename, distance, orb_match_count, combined_score) in enumerate(results, 1):
        plt.subplot(3, num_matches + 1, i + 1)
        plt.imshow(cv2.cvtColor(dataset_images[filename], cv2.COLOR_BGR2RGB))
        plt.title(f'Match {i}')
        plt.axis('off')

        metrics_text = (
            f'Color Distance: {distance:.3f}\n'
            f'ORB Matches: {orb_match_count}\n'
        )
        combined_score = (
            f'Combined Score: {combined_score:.3f}'
        )

        plt.subplot(3, num_matches + 1, num_matches + i + 2)
        plt.text(0.5, 0.65, metrics_text, ha='center', va='center', fontsize=10)  
        plt.text(0.5, 0.35, combined_score, ha='center', va='center', fontsize=9, fontweight='bold')  
        plt.axis('off')


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """
    Command line interface.
    """
    parser = argparse.ArgumentParser(description='Hybrid ORB and Color Matching')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--query', type=str, required=True, help='Path to the query image')
    parser.add_argument('--descriptors', type=str, required=True, help='Directory with ORB descriptor files')
    parser.add_argument('--top_orb', type=int, default=10, help='Number of top ORB matches to use (default: 15)')
    parser.add_argument('--top_color', type=int, default=5, help='Number of top color matches to use (default: 6)')

    args = parser.parse_args()

    hybrid_matching(args.dataset, args.query, args.descriptors, top_n_orb=args.top_orb, top_n_color=args.top_color)
