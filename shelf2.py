import cv2
import numpy as np
import easyocr
from difflib import SequenceMatcher
import os

class BakeryDisplayDetector:
    def __init__(self):
        print("Initializing EasyOCR...")
        self.reader = easyocr.Reader(['en'])
        print("Initialization complete!")
    
    def similar(self, a, b):
        """Check similarity between two strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def detect_display_sections(self, image):
        """Detect upper and lower display sections"""
        height, width = image.shape[:2]
        mid_height = height // 2
        
        return [
            {
                'image': image[0:mid_height, :],
                'name': 'Upper',
                'y': 0,
                'h': mid_height,
                'w': width
            },
            {
                'image': image[mid_height:, :],
                'name': 'Lower',
                'y': mid_height,
                'h': height - mid_height,
                'w': width
            }
        ]
    
    def process_grid_cell(self, cell_img, is_label_row=False):
        """Process a single grid cell using EasyOCR"""
        if cell_img.size == 0:
            return None
            
        try:
            # Detect text with EasyOCR
            results = self.reader.readtext(cell_img)
            
            if results:
                # Get the text with highest confidence
                best_result = max(results, key=lambda x: x[2])
                bbox, text, conf = best_result
                
                return {
                    'text': text,
                    'confidence': conf * 100,
                    'bbox': bbox,
                    'is_label': is_label_row
                }
        except Exception as e:
            print(f"Error processing cell: {e}")
            
        return None
    
    def process_section_grid(self, section):
        """Process a section as a 6x8 grid and associate labels with items"""
        image = section['image']
        height, width = image.shape[:2]
        
        # Create 6x8 grid
        rows, cols = 6, 8
        cell_height = height // rows
        cell_width = width // cols
        
        # Initialize grid for text detection
        text_grid = [[None for _ in range(cols)] for _ in range(rows)]
        item_label_pairs = []
        
        # Process each cell
        for i in range(rows):
            for j in range(cols):
                cell_x = j * cell_width
                cell_y = i * cell_height
                cell_img = image[cell_y:cell_y+cell_height, cell_x:cell_x+cell_width]
                
                # Check if this is a label row (every other row)
                is_label_row = (i % 2 == 1)
                cell_result = self.process_grid_cell(cell_img, is_label_row)
                text_grid[i][j] = cell_result
                
                # If this is a label, associate it with the item above
                if cell_result and is_label_row and i > 0:
                    item_cell = text_grid[i-1][j]
                    if item_cell:
                        item_label_pairs.append({
                            'item_cell': item_cell,
                            'label_cell': cell_result,
                            'position': (j, i-1),
                            'grid_pos': {'x': cell_x, 'y': cell_y - cell_height}
                        })
        
        return text_grid, (cell_height, cell_width), item_label_pairs
    
    def visualize_results(self, image_path, output_path="bakery_analysis.jpg"):
        """Visualize detected items and their labels"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        result = image.copy()
        height, width = image.shape[:2]
        
        # Create larger image with text panel
        text_panel_width = 400
        combined_image = np.zeros((height, width + text_panel_width, 3), dtype=np.uint8)
        combined_image[:, :width] = result
        combined_image[:, width:] = (255, 255, 255)
        
        sections = self.detect_display_sections(image)
        text_y_offset = 30
        
        colors = [(0, 255, 0), (0, 0, 255)]  # Green for upper, Red for lower
        
        for idx, section in enumerate(sections):
            color = colors[idx]
            
            # Process grid
            text_grid, (cell_height, cell_width), item_label_pairs = self.process_section_grid(section)
            
            # Add section header
            cv2.putText(combined_image, f"{section['name']} SECTION:", 
                       (width + 10, text_y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            text_y_offset += 30
            
            # Draw item-label pairs
            for pair in item_label_pairs:
                item_x = pair['grid_pos']['x']
                item_y = pair['grid_pos']['y'] + section['y']
                
                # Draw box around item and label
                cv2.rectangle(result, 
                            (item_x, item_y), 
                            (item_x + cell_width, item_y + cell_height * 2), 
                            color, 2)
                
                # Add to text panel
                item_text = pair['item_cell']['text']
                label_text = pair['label_cell']['text']
                text_info = f"  Item: {item_text} - Label: {label_text}"
                cv2.putText(combined_image, text_info,
                           (width + 20, text_y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                text_y_offset += 20
                
                # Draw text on image
                cv2.putText(result, f"{item_text}",
                           (item_x + 2, item_y + cell_height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                cv2.putText(result, f"{label_text}",
                           (item_x + 2, item_y + cell_height + cell_height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            text_y_offset += 20
        
        # Copy result back to combined image
        combined_image[:, :width] = result
        
        # Save and display
        print(f"Saving result to: {output_path}")
        cv2.imwrite(output_path, combined_image)
        cv2.imshow('Bakery Display Analysis', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        image_path = "bakery_display.jpg"
        detector = BakeryDisplayDetector()
        detector.visualize_results(image_path)
    except Exception as e:
        print(f"Error: {e}")