import cv2
import numpy as np
import pytesseract
from PIL import Image
from difflib import SequenceMatcher
import easyocr

def similar(a, b):
    """Check similarity between two strings"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def process_grid_cell(cell_img):
    """Process cell using EasyOCR"""
    PRODUCTS = [
        "plain nankhatai",
        "curry cookies",
        "salted banana chips",
        "spicy banana chips"
    ]
    
    # Initialize EasyOCR (do this once globally in practice)
    reader = easyocr.Reader(['en'])
    
    # Convert to RGB for EasyOCR
    if len(cell_img.shape) == 2:
        cell_img = cv2.cvtColor(cell_img, cv2.COLOR_GRAY2RGB)
    
    best_match = None
    best_confidence = 0
    
    try:
        # Detect text with EasyOCR
        results = reader.readtext(cell_img)
        
        for (bbox, text, conf) in results:
            # Check against each product name
            for product in PRODUCTS:
                similarity = similar(text, product)
                if similarity > 0.6:
                    confidence = similarity * conf * 100
                    if confidence > best_confidence:
                        best_match = {
                            'text': product,
                            'detected_text': text,
                            'confidence': confidence,
                            'bbox': bbox
                        }
                        best_confidence = confidence
    except:
        pass
    
    return best_match

def merge_neighboring_text(grid, cell_coords):
    """Merge text from neighboring cells to form complete product names"""
    rows, cols = 6, 8
    merged_text = []
    visited = set()
    
    def get_neighbors(r, c):
        """Get valid neighboring cell coordinates"""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                new_r, new_c = r + dr, c + dc
                if (0 <= new_r < rows and 0 <= new_c < cols and 
                    (new_r, new_c) not in visited and 
                    grid[new_r][new_c] is not None):
                    neighbors.append((new_r, new_c))
        return neighbors
    
    def merge_cluster(r, c):
        """Recursively merge text cluster"""
        if (r, c) in visited:
            return None
        visited.add((r, c))
        
        current_cell = grid[r][c]
        if not current_cell:
            return None
        
        # Start with the current cell's product name
        product_name = current_cell['text']
        confidence = current_cell['confidence']
        
        # Check neighbors for parts of the same product name
        for nr, nc in get_neighbors(r, c):
            neighbor_result = merge_cluster(nr, nc)
            if neighbor_result and neighbor_result['text'] == product_name:
                confidence = max(confidence, neighbor_result['confidence'])
        
        return {
            'text': product_name,
            'confidence': confidence
        }
    
    # Find product name clusters
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] and (r, c) not in visited:
                cluster_result = merge_cluster(r, c)
                if cluster_result:
                    merged_text.append({
                        'text': cluster_result['text'],
                        'cells': list(visited - set(cell_coords)),
                        'confidence': cluster_result['confidence']
                    })
                cell_coords = set(visited)
    
    return merged_text

def process_containers(image_path):
    # Read the image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Define container regions (2 rows x 3 columns)
    mid_height = height // 2
    third_width = width // 3
    
    regions = [
        # Top row
        {'x': 0, 'y': 0, 'w': third_width, 'h': mid_height, 'name': 'Top Left'},
        {'x': third_width, 'y': 0, 'w': third_width, 'h': mid_height, 'name': 'Top Center'},
        {'x': 2 * third_width, 'y': 0, 'w': third_width, 'h': mid_height, 'name': 'Top Right'},
        # Bottom row
        {'x': 0, 'y': mid_height, 'w': third_width, 'h': mid_height, 'name': 'Bottom Left'},
        {'x': third_width, 'y': mid_height, 'w': third_width, 'h': mid_height, 'name': 'Bottom Center'},
        {'x': 2 * third_width, 'y': mid_height, 'w': third_width, 'h': mid_height, 'name': 'Bottom Right'}
    ]
    
    # Process each region
    for region in regions:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        region_img = image[y:y+h, x:x+w]
        
        # Create 6x8 grid
        cell_height = h // 6
        cell_width = w // 8
        
        # Initialize grid for text detection
        text_grid = [[None for _ in range(8)] for _ in range(6)]
        
        # Process each cell
        for i in range(6):
            for j in range(8):
                cell_x = j * cell_width
                cell_y = i * cell_height
                cell_img = region_img[cell_y:cell_y+cell_height, cell_x:cell_x+cell_width]
                
                text_grid[i][j] = process_grid_cell(cell_img)
        
        # Store the text grid in the region
        region['text_grid'] = text_grid
        
        # Merge neighboring text
        region['text_clusters'] = merge_neighboring_text(text_grid, set())
        region['grid_dims'] = (cell_height, cell_width)
    
    return regions

def visualize_results(image_path, regions, output_path="product_detection.jpg"):
    # Read the image
    image = cv2.imread(image_path)
    result = image.copy()
    
    # Create larger image with text panel
    height, width = image.shape[:2]
    text_panel_width = 400
    combined_image = np.zeros((height, width + text_panel_width, 3), dtype=np.uint8)
    combined_image[:, :width] = result
    combined_image[:, width:] = (255, 255, 255)  # White background for text panel
    
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255)   # Yellow
    ]
    
    text_y_offset = 30  # Starting position for text panel
    
    # Draw regions and detected text
    for i, region in enumerate(regions):
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        color = colors[i % len(colors)]
        
        # Draw region boundary
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        
        # Add region name to text panel
        cv2.putText(combined_image, f"{region['name']}:", 
                   (width + 10, text_y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        text_y_offset += 30
        
        # Draw grid and cell text
        cell_h, cell_w = region['grid_dims']
        for row in range(6):
            for col in range(8):
                cell_x = x + col * cell_w
                cell_y = y + row * cell_h
                
                # Draw grid cell
                cv2.rectangle(result, 
                            (cell_x, cell_y), 
                            (cell_x + cell_w, cell_y + cell_h), 
                            color, 1)
                
                # Get text for this cell if it exists
                if region.get('text_grid'):
                    cell_data = region['text_grid'][row][col]
                    if cell_data:
                        # Draw product name with confidence
                        text = f"{cell_data['text']} ({cell_data['confidence']:.1f}%)"
                        cv2.putText(result, text[:15] + ('...' if len(text) > 15 else ''),
                                  (cell_x + 2, cell_y + cell_h//2),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                        
                        # Add to text panel
                        cell_info = f"  Cell [{row},{col}]: {cell_data['text']} ({cell_data['confidence']:.1f}%)"
                        cv2.putText(combined_image, cell_info,
                                  (width + 20, text_y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        text_y_offset += 20
        
        # Draw merged text clusters
        for cluster in region['text_clusters']:
            cells = cluster['cells']
            if cells:
                # Calculate cluster center
                center_y = y + sum(r * cell_h for r, _ in cells) // len(cells)
                center_x = x + sum(c * cell_w for _, c in cells) // len(cells)
                
                # Draw merged text
                merged_text = f"Merged: {cluster['text']}"
                cv2.putText(combined_image, merged_text,
                           (width + 20, text_y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                text_y_offset += 20
        
        text_y_offset += 20  # Extra space between regions
    
    # Copy the updated result back to combined image
    combined_image[:, :width] = result
    
    # Save and display
    cv2.imwrite(output_path, combined_image)
    cv2.imshow('Grid Text Analysis', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print detailed results
    print("\nDetailed Grid Text Analysis:")
    for region in regions:
        print(f"\n{region['name']}:")
        if region.get('text_grid'):
            print("Individual Cells:")
            for row in range(6):
                for col in range(8):
                    text = region['text_grid'][row][col]
                    if text:
                        print(f"Cell [{row},{col}]: {text}")
        print("\nMerged Clusters:")
        for i, cluster in enumerate(region['text_clusters']):
            print(f"Cluster {i+1}: {cluster['text']}")

# Usage
if __name__ == "__main__":
    image_path = "reference_im.jpg"
    regions = process_containers(image_path)
    visualize_results(image_path, regions, "product_detection.jpg")
