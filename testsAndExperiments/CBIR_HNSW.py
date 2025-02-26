import cv2
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq
import faiss

class ColorMatcher:
    """
    Class for performing color-based image matching using a combination of
    Conventional Colour Histogram (CCH) and Stacked Colour Histogram (SCH).
    """
    
    def __init__(self, bins=64, filter_size=7, iterations=5):
        self.bins = bins  
        self.filter_size = filter_size  
        self.iterations = iterations 

    def compute_cch(self, image):
        """
        Compute Conventional Colour Histogram (CCH) in RGB space.
        """
        image = cv2.resize(image, (512, 512))  
        hist = cv2.calcHist([image], [0, 1, 2], None, [
                            self.bins] * 3, [0, 256] * 3)
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def compute_sch(self, image):
        """
        Compute Stacked Colour Histogram (SCH) using mean filtering.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        sch_hist = np.zeros((self.bins,), dtype=np.float32)
        for _ in range(self.iterations):
            image = cv2.blur(image, (self.filter_size, self.filter_size))
            hist = cv2.calcHist([image], [0], None, [self.bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            sch_hist += hist  
        return sch_hist

    def compute_features(self, image):
        """
        Combine CCH and SCH into a single feature vector.
        """
        cch = self.compute_cch(image)
        sch = self.compute_sch(image)
        return np.concatenate((cch, sch))

    def compute_distance(self, features1, features2, alpha=0.5):
        """
        Compute weighted similarity between two feature vectors.
        """
        split_idx = len(features1) // 2
        cch1, sch1 = features1[:split_idx], features1[split_idx:]
        cch2, sch2 = features2[:split_idx], features2[split_idx:]

        cch_dist = cv2.compareHist(cch1.astype(
            np.float32), cch2.astype(np.float32), cv2.HISTCMP_CORREL)
        sch_dist = cv2.compareHist(sch1.astype(
            np.float32), sch2.astype(np.float32), cv2.HISTCMP_CORREL)

        return alpha * (1 - cch_dist) + (1 - alpha) * (1 - sch_dist)

    def find_matches(self, query_features, dataset_features, top_n=5):
        """
        Find the top-N best matches from the dataset based on color similarity.
        """
        distances = [(filename, self.compute_distance(query_features, features))
                     for filename, features in dataset_features.items()]
        return sorted(distances, key=lambda x: x[1])[:top_n]


class HNSWOrbMatcher:
    """
    ORB Matcher using HNSW (Hierarchical Navigable Small World) with Faiss.
    """
    def __init__(self, descriptor_dim=32, ef_search=128, m=32):
        """
        descriptor_dim: ORB descriptor dimension (always 32)
        ef_search: Number of candidate neighbors to examine
        m: Number of bi-directional links per node
        """
        self.descriptor_dim = descriptor_dim
        self.index = faiss.IndexHNSWFlat(self.descriptor_dim, m)
        self.index.hnsw.efSearch = ef_search
        self.orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, edgeThreshold=31)
        self.image_lookup = []  
        self.query_descriptors = 500
        self.query_keypoints = 2000

    def get_descriptors(self):
        return self.query_descriptors
    
    def add_descriptors(self, descriptor_dir):
        """
        Loads ORB descriptors from a dataset and builds the HNSW index.
        """
        all_descriptors = []
        image_id = 0

        for filename in os.listdir(descriptor_dir):
            if filename.endswith('.pkl'):
                _, descriptors = self.load_descriptors(os.path.join(descriptor_dir, filename))

                if descriptors is not None and descriptors.shape[0] > 0:
                    all_descriptors.append(descriptors)
                    self.image_lookup.append(filename.replace('.pkl', '.jpg'))  # Store filenames
                    image_id += 1

        if all_descriptors:
            all_descriptors = np.vstack(all_descriptors).astype(np.float32)   # Add this line 
            self.index.add(all_descriptors)  

    def search(self, query_image_path, top_k):
        """
        Search for the top_k best matches using HNSW.
        """
        _, self.query_descriptors = self.generate_orb_descriptors(query_image_path)

        if self.query_descriptors.shape[0] == 0:
            return []

        distances, indices = self.index.search(self.query_descriptors, top_k)

        results = []
        for i in range(len(indices)):
            if indices[i][0] != -1 and indices[i][0] < len(self.image_lookup):
                results.append((self.image_lookup[indices[i][0]], distances[i][0]))

        return sorted(results, key=lambda x: x[1])


    def load_descriptors(self, descriptor_file):
        """
        Load ORB descriptors from a .pkl file.
        """
        with open(descriptor_file, 'rb') as f:
            keypoints, descriptors = pickle.load(f)
        return keypoints, descriptors.astype(np.float32) if descriptors is not None else None

    def generate_orb_descriptors(self, image_path):
        """
        Compute ORB keypoints and descriptors for an image.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Error: Could not load image {image_path}.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None:
            return [], np.array([], dtype=np.float32)

        return keypoints, descriptors.astype(np.float32) 


    def save_orb_descriptors(self, image_path, output_filename):
        """
        Compute ORB descriptors and save them to a file.
        """
        keypoints, descriptors = self.generate_orb_descriptors(image_path)
        with open(output_filename, 'wb') as f:
            pickle.dump((keypoints, descriptors), f)  

    


def hybrid_matching(dataset_path, query_image_path, descriptor_dir, top_n_orb=10, top_n_color=6):
    """
    Hybrid image matching using:
    - HNSW (for ORB feature matching) for fast approximate nearest neighbors.
    - Color Histograms for fine-tuned ranking.
    - Combines both scores for ranking.
    """
    orb_matcher = HNSWOrbMatcher()
    color_matcher = ColorMatcher()

    orb_matcher.add_descriptors(descriptor_dir)
    orb_matches = orb_matcher.search(query_image_path, top_n_orb)
    
    query_image = cv2.imread(query_image_path)
    query_features = color_matcher.compute_features(query_image)

    dataset_features = defaultdict(list)
    dataset_images = defaultdict(list)

    results = [] 
    orb_matches_dict = {filename: match_count for filename, match_count in orb_matches}

    for filename, _ in orb_matches:
        image_path = os.path.join(
            dataset_path, filename.replace('.pkl', '.jpg'))
        image = cv2.imread(image_path)
        dataset_images[filename] = image
        dataset_features[filename] = color_matcher.compute_features(image)

    color_matches = color_matcher.find_matches(
        query_features, dataset_features, top_n=top_n_color)

    weight_color = 0.15  
    weight_orb = 0.85 
    max_orb_match_count = len(orb_matcher.get_descriptors()) 
    
    for filename, distance in color_matches:
        orb_match_count = orb_matches_dict.get(filename, 0) 
        
        normalized_orb_match_count = orb_match_count / max_orb_match_count 
        
        if weight_color == 0 or distance == 0:
            combined_score = float('inf')  
        else:
            combined_score = (weight_orb * normalized_orb_match_count) / \
                (weight_color * distance)
        
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
