import cv2
import numpy as np
import os
from scipy.spatial.distance import euclidean
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import find_contours
from scipy.optimize import linear_sum_assignment

def resize_image(image, size=(100, 100)):
    """Resizes the image to a consistent size."""
    return cv2.resize(image, size)

def extract_color_histogram(image):
    """Extract color histogram features."""
    # Convert to grayscale if color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist.flatten()

def extract_texture(image):
    """Extract texture features using Gray Level Co-occurrence Matrix (GLCM)."""
    # Convert to grayscale if color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute GLCM
    glcm = graycomatrix(image, [1], [0], levels=256, symmetric=True, normed=True)
    
    # Extract texture properties
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])

def extract_shape(image):
    """Extract shape features."""
    # Convert to grayscale if color image
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find contours
    contours = find_contours(image, 0.8)
    
    if len(contours) == 0:
        return np.zeros(5)  # Return zero features if no contours
    
    # Take the largest contour
    largest_contour = max(contours, key=len)
    
    # Extract shape features
    perimeter = len(largest_contour)
    area = len(largest_contour)
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    return np.array([
        perimeter,  # Contour length
        area,       # Contour area
        compactness,# Shape compactness
        np.mean(largest_contour[:, 0]),  # Mean X
        np.mean(largest_contour[:, 1])   # Mean Y
    ])

def extract_features(image):
    """Extract features from an image region."""
    # Resize for consistent feature extraction
    resized = cv2.resize(image, (100, 100))
    
    # Convert to HSV for better color features
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    # Calculate color histogram
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Calculate texture features using grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # Combine features
    return np.concatenate([hist, [contrast, homogeneity]])

def segment_cupcakes(image):
    """Segments individual cupcakes in the collection image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cupcake_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:  # Ignore small contours
            cupcake_regions.append((x, y, w, h))
    return cupcake_regions

def match_cupcakes(collection_image, labeled_images, similarity_threshold=1.5):
    """
    Matches segmented cupcakes to labeled images with improved multi-tart handling.
    
    Args:
    - collection_image: Path to the main image with multiple tarts
    - labeled_images: Dictionary of reference tart images
    - similarity_threshold: Maximum euclidean distance to consider a match
    """
    collection = cv2.imread(collection_image)
    cupcake_regions = segment_cupcakes(collection)
    
    # Compute features for all reference tarts
    labeled_features = {}
    for label, image_path in labeled_images.items():
        image = cv2.imread(image_path)
        labeled_features[label] = extract_features(image)
    
    # Track used labels to prevent duplicate matches
    used_labels = set()
    matches = []
    
    for (x, y, w, h) in cupcake_regions:
        cupcake = collection[y:y+h, x:x+w]
        cupcake_features = extract_features(cupcake)
        
        best_match = None
        min_distance = float("inf")
        
        # Find best match among unused labels
        for label, features in labeled_features.items():
            if label in used_labels:
                continue
            
            try:
                distance = euclidean(cupcake_features, features)
                if distance < min_distance and distance < similarity_threshold:
                    min_distance = distance
                    best_match = label
            except ValueError as e:
                print(f"Error comparing {label}: {e}")
        
        # If a match is found, add it and mark the label as used
        if best_match:
            matches.append({
                "label": best_match, 
                "position": (x, y, w, h),
                "distance": min_distance
            })
            used_labels.add(best_match)
    
    # Print matching details
    print(f"Total tarts found: {len(matches)}")
    for match in matches:
        print(f"Matched {match['label']} at position {match['position']} with distance {match['distance']:.2f}")
    
    return matches

def create_boundary_boxes(image_path):
    """Create bounding boxes for tarts using color and contour detection."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        return []
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define green color range (for banana leaf background)
    lower_green = np.array([35, 20, 20])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for non-green areas (tarts)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    tart_mask = cv2.bitwise_not(green_mask)
    
    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    tart_mask = cv2.morphologyEx(tart_mask, cv2.MORPH_CLOSE, kernel)
    tart_mask = cv2.morphologyEx(tart_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(
        tart_mask, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter and sort contours by area
    min_area = 1000  # Adjust based on your image size
    max_area = 50000  # Adjust based on your image size
    
    boundary_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            boundary_boxes.append((x, y, w, h))
    
    # Sort boxes from top to bottom, left to right
    boundary_boxes.sort(key=lambda box: (box[1], box[0]))
    
    # Create a copy of the image for visualization
    result_image = image.copy()
    
    # Draw bounding boxes with numbers
    for i, (x, y, w, h) in enumerate(boundary_boxes):
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add tart number
        cv2.putText(result_image, 
                    f'Tart {i+1}', 
                    (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2)
    
    # Save results
    output_path = f"output_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, result_image)
    
    # Save debug mask
    cv2.imwrite(f"debug_mask_{os.path.basename(image_path)}", tart_mask)
    
    return boundary_boxes

def detect_non_green_clusters(image_path):
    """Detect and cluster non-green colors in the image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define green color range to exclude (adjusted for better detection)
    lower_green = np.array([35, 20, 20])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for non-green areas
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    non_green_mask = cv2.bitwise_not(green_mask)
    
    # Clean up mask using morphological operations
    kernel = np.ones((7,7), np.uint8)  # Increased kernel size
    cleaned_mask = cv2.morphologyEx(non_green_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter and sort contours by area
    min_area = 1000  # Adjusted minimum area
    max_area = 100000  # Adjusted maximum area
    valid_clusters = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            valid_clusters.append((x, y, w, h))
    
    # Sort clusters by vertical position (top to bottom)
    valid_clusters.sort(key=lambda x: x[1])
    
    # Debug print
    print(f"Found {len(valid_clusters)} clusters before filtering")
    
    # Ensure exactly 5 clusters by adjusting area threshold if needed
    if len(valid_clusters) != 5:
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
    
    # Save debug images
    cv2.imwrite("clusters_mask.jpg", cleaned_mask)
    cv2.imwrite("clusters_visual.jpg", visualization)
    
    return valid_clusters

def match_clusters_to_references(image_path, reference_paths):
    """Match detected clusters to reference tart images."""
    output_dir = f"results_{image_path.split('.')[0]}"
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    # Detect clusters
    clusters = detect_non_green_clusters(image_path)
    if not clusters:
        print("No clusters detected")
        return
    
    # Load reference images and extract features
    reference_features = {}
    for ref_name, ref_path in reference_paths.items():
        ref_img = cv2.imread(ref_path)
        if ref_img is not None:
            reference_features[ref_name] = extract_features(ref_img)
    
    # Extract features from each cluster
    cluster_features = []
    for (x, y, w, h) in clusters:
        cluster_region = image[y:y+h, x:x+w]
        features = extract_features(cluster_region)
        cluster_features.append((features, (x, y, w, h)))
    
    # Create cost matrix
    n = len(reference_features)  # Should be 5
    cost_matrix = np.zeros((n, n))
    
    # Fill cost matrix with feature distances
    for i, (cluster_feat, _) in enumerate(cluster_features):
        for j, ref_feat in enumerate(reference_features.values()):
            cost_matrix[i, j] = np.linalg.norm(cluster_feat - ref_feat)
    
    # Handle missing clusters if any
    if len(clusters) < len(reference_features):
        # Pad cost matrix with large values
        pad_rows = len(reference_features) - len(clusters)
        padding = np.full((pad_rows, len(reference_features)), 1000.0)
        cost_matrix = np.vstack([cost_matrix, padding])
    
    # Find optimal matching
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError as e:
        print(f"Error in matching: {e}")
        return []
    
    # Create visualization
    result_image = image.copy()
    matches = []
    
    # Draw matches
    for cluster_idx, ref_idx in zip(row_ind, col_ind):
        if cluster_idx < len(cluster_features):  # Only draw actual clusters
            _, (x, y, w, h) = cluster_features[cluster_idx]
            ref_name = list(reference_features.keys())[ref_idx]
            matches.append((ref_name, (x, y, w, h)))
            
            # Draw rectangle and label
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_image, 
                        ref_name,
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)
    
    # Save results with proper paths
    result_path = os.path.join(output_dir, "matched_clusters.jpg")
    cv2.imwrite(result_path, result_image)
    
    return matches

def compare_tart_arrangements(tart1_matches, tart3_matches, image_path):
    """Compare tart arrangements between tart1 (admin) and tart3."""
    # Load tart3 image for visualization
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image {image_path}")
        return
    
    # Sort matches by vertical position (top to bottom) and then horizontal position (left to right)
    tart1_sorted = sorted(tart1_matches, key=lambda x: (x[1][1], x[1][0]))
    tart3_sorted = sorted(tart3_matches, key=lambda x: (x[1][1], x[1][0]))
    
    # Create visualization
    result_image = image.copy()
    matches_correct = []
    
    # Compare each position
    for (ref1, pos1), (ref3, pos3) in zip(tart1_sorted, tart3_sorted):
        x, y, w, h = pos3
        # Check if labels match for this position
        matches = ref1 == ref3
        matches_correct.append(matches)
        
        # Choose color based on match (green for match, red for mismatch)
        color = (0, 255, 0) if matches else (0, 0, 255)
        
        # Draw rectangle and label
        cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(result_image, 
                    f'{ref3}{"✓" if matches else "✗"}',
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2)
    
    # Add overall match status
    all_match = all(matches_correct)
    status_text = "All Positions Match!" if all_match else "Position Mismatches Found"
    status_color = (0, 255, 0) if all_match else (0, 0, 255)
    cv2.putText(result_image,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                status_color,
                2)
    
    # Save comparison result
    cv2.imwrite("tart_comparison_result.jpg", result_image)
    
    return all_match, matches_correct

def process_and_compare_images():
    """Process both images and compare their arrangements."""
    reference_paths = {
        "og_tart1": "og_tart1.jpg",
        "og_tart2": "og_tart2.jpg",
        "og_tart3": "og_tart3.jpg",
        "og_tart4": "og_tart4.jpg",
        "og_tart5": "og_tart5.jpg"
    }
    
    # Process tart1 (admin)
    print("\nProcessing tart1.jpg (admin)...")
    tart1_matches = match_clusters_to_references("tart1.jpg", reference_paths)
    
    # Process tart3
    print("\nProcessing tart3.jpg...")
    tart3_matches = match_clusters_to_references("tart3.jpg", reference_paths)
    
    if tart1_matches and tart3_matches:
        print("\nComparing arrangements...")
        all_match, position_matches = compare_tart_arrangements(
            tart1_matches, 
            tart3_matches,
            "tart3.jpg"
        )
        
        # Print detailed comparison results
        print("\nComparison Results:")
        print(f"Overall Match: {'✓' if all_match else '✗'}")
        for i, matches in enumerate(position_matches):
            print(f"Position {i+1}: {'✓' if matches else '✗'}")

# Run the processing and comparison
if __name__ == "__main__":
    process_and_compare_images()