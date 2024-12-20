import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from scipy.spatial.distance import euclidean
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import find_contours
from scipy.optimize import linear_sum_assignment

class TartVerifier:
    def __init__(self):
        print("[DEBUG] Initializing TartVerifier")
        self.admin_image = None
        self.user_image = None
        # Initialize reference paths
        self.reference_paths = {
            "og_tart1": "og_tart1.jpg",
            "og_tart2": "og_tart2.jpg",
            "og_tart3": "og_tart3.jpg",
            "og_tart4": "og_tart4.jpg",
            "og_tart5": "og_tart5.jpg"
        }
        print("[DEBUG] Reference paths initialized:", self.reference_paths)
    
    def set_admin_image(self, image_path):
        print(f"[DEBUG] Setting admin image: {image_path}")
        self.admin_image = image_path
        if not os.path.exists(image_path):
            print("[ERROR] Admin image path does not exist")
            raise ValueError("Admin image path does not exist")
        print("[DEBUG] Admin image set successfully")
    
    def set_user_image(self, image_path):
        print(f"[DEBUG] Setting user image: {image_path}")
        self.user_image = image_path
        if not os.path.exists(image_path):
            print("[ERROR] User image path does not exist")
            raise ValueError("User image path does not exist")
        print("[DEBUG] User image set successfully")
    
    def compare_images(self):
        print("[DEBUG] Starting image comparison")
        if not self.admin_image or not self.user_image:
            print("[ERROR] Missing admin or user image")
            return {'error': 'Both admin and user images are required'}
        
        try:
            print("[DEBUG] Matching admin image clusters to references")
            admin_matches = self.match_clusters_to_references(self.admin_image)
            print(f"[DEBUG] Admin matches: {admin_matches}")
            
            print("[DEBUG] Matching user image clusters to references")
            user_matches = self.match_clusters_to_references(self.user_image)
            print(f"[DEBUG] User matches: {user_matches}")
            
            if admin_matches and user_matches:
                print("[DEBUG] Comparing tart arrangements")
                all_match, position_matches = self.compare_tart_arrangements(
                    admin_matches,
                    user_matches,
                    self.user_image
                )
                print(f"[DEBUG] Comparison results - All match: {all_match}, Position matches: {position_matches}")
                
                return all_match, position_matches
            
            print("[DEBUG] No matches found, returning default false values")
            return False, [False] * 5
            
        except Exception as e:
            print(f"[ERROR] Error in compare_images: {str(e)}")
            return False, [False] * 5
    
    def detect_non_green_clusters(self, image_path):
        """Detect and cluster non-green colors in the image."""
        print(f"[DEBUG] Detecting non-green clusters in {image_path}")
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Cannot load image {image_path}")
            return []
        
        print("[DEBUG] Converting image to HSV")
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define green color range to exclude (adjusted for better detection)
        lower_green = np.array([35, 20, 20])
        upper_green = np.array([85, 255, 255])
        print(f"[DEBUG] Green color range - Lower: {lower_green}, Upper: {upper_green}")
        
        # Create mask for non-green areas
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        non_green_mask = cv2.bitwise_not(green_mask)
        
        print("[DEBUG] Applying morphological operations")
        # Clean up mask using morphological operations
        kernel = np.ones((7,7), np.uint8)  # Increased kernel size
        cleaned_mask = cv2.morphologyEx(non_green_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        print("[DEBUG] Finding contours")
        # Find contours
        contours, _ = cv2.findContours(
            cleaned_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        print(f"[DEBUG] Found {len(contours)} initial contours")
        
        # Filter and sort contours by area
        min_area = 1000  # Adjusted minimum area
        max_area = 100000  # Adjusted maximum area
        valid_clusters = []
        
        print("[DEBUG] Filtering contours by area")
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                valid_clusters.append((x, y, w, h))
        
        # Sort clusters by vertical position (top to bottom)
        valid_clusters.sort(key=lambda x: x[1])
        
        print(f"[DEBUG] Found {len(valid_clusters)} clusters before filtering")
        
        # Ensure exactly 5 clusters by adjusting area threshold if needed
        if len(valid_clusters) != 5:
            print("[DEBUG] Adjusting area threshold to get exactly 5 clusters")
            areas = [cv2.contourArea(cnt) for cnt in contours]
            areas.sort(reverse=True)
            if len(areas) >= 5:
                min_area = areas[4] - 100  # Set threshold just below the 5th largest area
                valid_clusters = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > min_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        valid_clusters.append((x, y, w, h))
                valid_clusters.sort(key=lambda x: x[1])
                valid_clusters = valid_clusters[:5]  # Take top 5
        
        print("[DEBUG] Creating visualization")
        # Create visualization
        visualization = image.copy()
        for i, (x, y, w, h) in enumerate(valid_clusters):
            cv2.rectangle(visualization, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(visualization, 
                       f'Cluster {i+1}',
                       (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (0, 255, 0),
                       2)
        
        print("[DEBUG] Saving debug images")
        # Save debug images
        cv2.imwrite("clusters_mask.jpg", cleaned_mask)
        cv2.imwrite("clusters_visual.jpg", visualization)
        
        print(f"[DEBUG] Returning {len(valid_clusters)} final clusters")
        return valid_clusters

    def extract_features(self, image):
        """Extract features from an image region."""
        print("[DEBUG] Extracting features from image region")
        # Resize for consistent feature extraction
        resized = cv2.resize(image, (100, 100))
        
        print("[DEBUG] Converting to HSV")
        # Convert to HSV for better color features
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        
        print("[DEBUG] Calculating color histogram")
        # Calculate color histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        print("[DEBUG] Calculating texture features")
        # Calculate texture features using grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        
        # Combine features
        features = np.concatenate([hist, [contrast, homogeneity]])
        print(f"[DEBUG] Extracted {len(features)} features")
        return features

    def match_clusters_to_references(self, image_path):
        """Match detected clusters to reference tart images."""
        print(f"[DEBUG] Matching clusters to references for {image_path}")
        # Load and process main image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Cannot load image {image_path}")
            return []
        
        print("[DEBUG] Detecting clusters")
        # Detect clusters
        clusters = self.detect_non_green_clusters(image_path)
        if not clusters:
            print("[DEBUG] No clusters detected")
            return []
        
        print("[DEBUG] Loading reference images")
        # Load reference images and extract features
        reference_features = {}
        for ref_name, ref_path in self.reference_paths.items():
            print(f"[DEBUG] Processing reference {ref_name}")
            ref_img = cv2.imread(ref_path)
            if ref_img is not None:
                reference_features[ref_name] = self.extract_features(ref_img)
        
        print("[DEBUG] Extracting features from clusters")
        # Extract features from each cluster
        cluster_features = []
        for (x, y, w, h) in clusters:
            cluster_region = image[y:y+h, x:x+w]
            features = self.extract_features(cluster_region)
            cluster_features.append((features, (x, y, w, h)))
        
        print("[DEBUG] Creating cost matrix")
        # Create cost matrix
        n = len(reference_features)  # Should be 5
        cost_matrix = np.zeros((n, n))
        
        # Fill cost matrix with feature distances
        for i, (cluster_feat, _) in enumerate(cluster_features):
            for j, ref_feat in enumerate(reference_features.values()):
                cost_matrix[i, j] = np.linalg.norm(cluster_feat - ref_feat)
        
        print("[DEBUG] Finding optimal matching")
        # Find optimal matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create matches list
        matches = []
        for cluster_idx, ref_idx in zip(row_ind, col_ind):
            if cluster_idx < len(cluster_features):
                _, pos = cluster_features[cluster_idx]
                ref_name = list(reference_features.keys())[ref_idx]
                matches.append((ref_name, pos))
        
        print(f"[DEBUG] Found {len(matches)} matches")
        return matches

    def compare_tart_arrangements(self, admin_matches, user_matches, user_image_path):
        """Compare tart arrangements between admin and user images."""
        print("[DEBUG] Comparing tart arrangements")
        # Load user image for visualization
        image = cv2.imread(user_image_path)
        if image is None:
            print(f"[ERROR] Cannot load image {user_image_path}")
            return False, []
        
        print("[DEBUG] Sorting matches by position")
        # Sort matches by vertical position (top to bottom) and then horizontal position (left to right)
        admin_sorted = sorted(admin_matches, key=lambda x: (x[1][1], x[1][0]))
        user_sorted = sorted(user_matches, key=lambda x: (x[1][1], x[1][0]))
        
        print("[DEBUG] Creating visualization")
        # Create visualization
        result_image = image.copy()
        matches_correct = []
        
        print("[DEBUG] Comparing positions")
        # Compare each position
        for (ref_admin, pos_admin), (ref_user, pos_user) in zip(admin_sorted, user_sorted):
            x, y, w, h = pos_user
            # Check if labels match for this position
            matches = ref_admin == ref_user
            matches_correct.append(matches)
            
            # Choose color based on match (green for match, red for mismatch)
            color = (0, 255, 0) if matches else (0, 0, 255)
            
            # Draw rectangle and label with match indicator
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result_image, 
                        f'{ref_user} {"✓" if matches else "✗"}',
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2)
        
        # Add overall match status
        all_match = all(matches_correct)
        status_text = "All Positions Match!" if all_match else "Position Mismatches Found"
        cv2.putText(result_image,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if all_match else (0, 0, 255),
                    2)
        
        print("[DEBUG] Saving comparison result")
        # Save comparison result
        cv2.imwrite("static/tart_comparison_result.jpg", result_image)
        
        print(f"\n[DEBUG] Comparison Results:")
        print(f"[DEBUG] Overall Match: {'✓' if all_match else '✗'}")
        for i, (matches, ((ref_admin, _), (ref_user, _))) in enumerate(zip(matches_correct, zip(admin_sorted, user_sorted))):
            print(f"[DEBUG] Position {i+1}: {'✓' if matches else '✗'} (Admin: {ref_admin} vs User: {ref_user})")
        
        return all_match, matches_correct