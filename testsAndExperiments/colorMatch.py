import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import time


    
class SmartColorMatcher:
    """
    Class for performing color-based image matching using HSV histograms.
    It computes both global and local color histograms for robust matching.
    """
    def __init__(self):
        self.bins = (8, 12, 3)  
        
    def compute_features(self, image):
        """Compute color and edge features efficiently."""
        
        image = cv2.resize(image, (256, 256))

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([hsv], [0, 1], None, [self.bins[0], self.bins[1]],
                            [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [50], [0, 256])
        cv2.normalize(edge_hist, edge_hist, alpha=0,
                      beta=1, norm_type=cv2.NORM_MINMAX)

        return np.concatenate([hist.flatten(), edge_hist.flatten()])

    def match_images(self, query_features, dataset_features):
        """Match features using correlation."""
        distances = []
        for filename, features in dataset_features.items():
            correlation = np.correlate(query_features, features)[0]
            distances.append((filename, correlation))

        return sorted(distances, key=lambda x: x[1], reverse=True)


def process_dataset(dataset_path, matcher, max_images=500):
    """Process dataset with progress tracking and timing."""
    images = {}
    features = {}
    count = 0
    start_time = time.time()

    print("\nLoading and processing dataset images...")
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                path = os.path.join(dataset_path, filename)
                image = cv2.imread(path)

                if image is not None:
                    images[filename] = image
                    features[filename] = matcher.compute_features(image)
                    count += 1

                    if count % 50 == 0:
                        elapsed = time.time() - start_time
                        print(
                            f"Processed {count} images... ({elapsed:.2f} seconds)")

                    if count >= max_images:
                        break

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    total_time = time.time() - start_time
    print(f"\nFinished processing {count} images in {total_time:.2f} seconds")
    return images, features


def main():
    parser = argparse.ArgumentParser(description='Smart Color Matching')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset folder')
    parser.add_argument('--query', type=str, required=True,
                        help='Path to query image')
    parser.add_argument('--max_images', type=int, default=500,
                        help='Maximum number of dataset images to process')
    args = parser.parse_args()

    try:
        matcher = SmartColorMatcher()
        print("\nInitialized SmartColorMatcher")

        # Load and process query image
        print(f"\nProcessing query image: {args.query}")
        query_image = cv2.imread(args.query)
        if query_image is None:
            raise ValueError("Could not load query image")
        query_features = matcher.compute_features(query_image)
        print("Query image processed successfully")

        # Process dataset
        dataset_images, dataset_features = process_dataset(
            args.dataset, matcher, args.max_images)

        if not dataset_images:
            raise ValueError("No images loaded from dataset")

        # Find matches
        print("\nFinding matches...")
        start_time = time.time()
        matches = matcher.match_images(query_features, dataset_features)
        match_time = time.time() - start_time
        print(f"Matching completed in {match_time:.2f} seconds")

        # Display results
        print("\nDisplaying top 5 matches:")
        plt.figure(figsize=(15, 5))

        # Show query image
        plt.subplot(1, 6, 1)
        plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
        plt.title('Query Image')
        plt.axis('off')

        # Show top 5 matches
        for i, (filename, score) in enumerate(matches[:5], 2):
            plt.subplot(1, 6, i)
            plt.imshow(cv2.cvtColor(
                dataset_images[filename], cv2.COLOR_BGR2RGB))
            plt.title(f'Match {i-1}\nScore: {score:.3f}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Print match scores
        print("\nTop 5 matches (filename: score):")
        for filename, score in matches[:5]:
            print(f"{filename}: {score:.4f}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == '__main__':
    main()
