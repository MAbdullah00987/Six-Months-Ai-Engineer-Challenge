     
#Image Manipulator:
#Load an image into a NumPy array and perform operations like grayscale conversion, cropping, and flipping.


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union


class ImageManipulator:
    """
    A class for performing various image manipulation operations.
    
    Attributes:
    -----------
    image : np.ndarray
        The current image stored as a NumPy array
    original_image : np.ndarray
        A copy of the original image for reference
    """
    
    def __init__(self):
        """Initialize the ImageManipulator with no image loaded."""
        self.image = None
        self.original_image = None
    
    
    # ==================== SECTION 1: IMAGE LOADING ====================
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file path and store it as a NumPy array.
        
        Parameters:
        -----------
        image_path : str
            Path to the image file (supports PNG, JPG, etc.)
            
        Returns:
        --------
        np.ndarray
            Loaded image as NumPy array with shape (H, W, C)
            
        Raises:
        -------
        FileNotFoundError
            If the image file doesn't exist
        """
        try:
            # Open image using Pillow's Image.open()
            # This creates a PIL Image object
            pil_image = Image.open(image_path)
            
            # Convert PIL Image to NumPy array
            # np.array() creates a copy of the image data
            # Data type is uint8 (0-255) by default
            self.image = np.array(pil_image)
            
            # Store a copy of the original for reference
            # copy() creates a deep copy, preventing modifications
            self.original_image = self.image.copy()
            
            # Print image information for debugging
            print("=" * 60)
            print("IMAGE LOADED SUCCESSFULLY")
            print("=" * 60)
            print(f"File path: {image_path}")
            print(f"Image shape: {self.image.shape}")
            print(f"  - Height: {self.image.shape[0]} pixels")
            print(f"  - Width: {self.image.shape[1]} pixels")
            
            # Check if image has color channels
            if len(self.image.shape) == 3:
                print(f"  - Channels: {self.image.shape[2]} (RGB)")
            else:
                print(f"  - Channels: 1 (Grayscale)")
            
            print(f"Data type: {self.image.dtype}")
            print(f"Value range: [{self.image.min()}, {self.image.max()}]")
            print(f"Memory size: {self.image.nbytes / 1024:.2f} KB")
            print("=" * 60)
            
            return self.image
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
    
    
    # ==================== SECTION 2: GRAYSCALE CONVERSION ====================
    
    def convert_to_grayscale(self) -> np.ndarray:
        """
        Convert the current RGB image to grayscale using luminosity method.
        
        The luminosity method uses weighted coefficients based on human perception:
        - Green: 0.587 (humans are most sensitive to green)
        - Red: 0.299 (moderate sensitivity)
        - Blue: 0.114 (least sensitivity)
        
        Formula: Gray = 0.299*R + 0.587*G + 0.114*B
        
        Returns:
        --------
        np.ndarray
            Grayscale image with shape (H, W)
            
        Raises:
        -------
        ValueError
            If no image is loaded or image is already grayscale
        """
        # Check if image is loaded
        if self.image is None:
            raise ValueError("No image loaded. Use load_image() first.")
        
        # Check if image is already grayscale
        if len(self.image.shape) == 2:
            print("Image is already in grayscale format.")
            return self.image
        
        # Define luminosity coefficients (ITU-R BT.601 standard)
        # These weights reflect human perception of brightness
        weights = np.array([0.299, 0.587, 0.114])
        
        # Perform weighted sum across color channels
        # np.dot() performs matrix multiplication:
        # - For each pixel (H×W), multiply RGB values by weights
        # - Sum the results: Gray = R*0.299 + G*0.587 + B*0.114
        # - Result shape: (H, W, 3) × (3,) → (H, W)
        grayscale = np.dot(self.image[..., :3], weights)
        
        # Convert to uint8 (0-255 integer values)
        # This is necessary for proper image display and saving
        grayscale = grayscale.astype(np.uint8)
        
        # Update the current image
        self.image = grayscale
        
        print("\n" + "=" * 60)
        print("GRAYSCALE CONVERSION COMPLETED")
        print("=" * 60)
        print(f"Original shape: {self.original_image.shape} (RGB)")
        print(f"New shape: {self.image.shape} (Grayscale)")
        print(f"Method: Luminosity (0.299R + 0.587G + 0.114B)")
        print("=" * 60)
        
        return self.image
    
    
    # ==================== SECTION 3: CROPPING OPERATIONS ====================
    
    def crop_center(self, crop_height: int, crop_width: int) -> np.ndarray:
        """
        Crop the center region of the image.
        
        Parameters:
        -----------
        crop_height : int
            Desired height of the cropped region
        crop_width : int
            Desired width of the cropped region
            
        Returns:
        --------
        np.ndarray
            Cropped image centered on the original
            
        Raises:
        -------
        ValueError
            If crop dimensions exceed image dimensions
        """
        if self.image is None:
            raise ValueError("No image loaded. Use load_image() first.")
        
        # Get current image dimensions
        # shape[:2] extracts (height, width), ignoring channels
        height, width = self.image.shape[:2]
        
        # Validate crop dimensions
        if crop_height > height or crop_width > width:
            raise ValueError(
                f"Crop dimensions ({crop_height}×{crop_width}) exceed "
                f"image dimensions ({height}×{width})"
            )
        
        # Calculate starting coordinates to center the crop
        # Integer division (//) ensures whole pixel values
        start_y = (height - crop_height) // 2
        start_x = (width - crop_width) // 2
        
        # Calculate ending coordinates
        end_y = start_y + crop_height
        end_x = start_x + crop_width
        
        # Perform array slicing to extract the crop
        # [start_y:end_y] - selects rows (vertical slice)
        # [start_x:end_x] - selects columns (horizontal slice)
        # [...] or [:] - keeps all remaining dimensions (channels)
        self.image = self.image[start_y:end_y, start_x:end_x]
        
        print("\n" + "=" * 60)
        print("CENTER CROP COMPLETED")
        print("=" * 60)
        print(f"Original size: {height}×{width}")
        print(f"Crop size: {crop_height}×{crop_width}")
        print(f"Crop region: ({start_x}, {start_y}) to ({end_x}, {end_y})")
        print("=" * 60)
        
        return self.image
    
    
    def crop_top_left(self, crop_height: int, crop_width: int) -> np.ndarray:
        """
        Crop from the top-left corner of the image.
        
        Parameters:
        -----------
        crop_height : int
            Desired height of the cropped region
        crop_width : int
            Desired width of the cropped region
            
        Returns:
        --------
        np.ndarray
            Cropped image from top-left corner
        """
        if self.image is None:
            raise ValueError("No image loaded. Use load_image() first.")
        
        height, width = self.image.shape[:2]
        
        # Validate crop dimensions
        if crop_height > height or crop_width > width:
            raise ValueError(
                f"Crop dimensions ({crop_height}×{crop_width}) exceed "
                f"image dimensions ({height}×{width})"
            )
        
        # Slice from (0,0) to (crop_height, crop_width)
        # [:crop_height] is equivalent to [0:crop_height]
        # [:crop_width] is equivalent to [0:crop_width]
        self.image = self.image[:crop_height, :crop_width]
        
        print("\n" + "=" * 60)
        print("TOP-LEFT CROP COMPLETED")
        print("=" * 60)
        print(f"Original size: {height}×{width}")
        print(f"Crop size: {crop_height}×{crop_width}")
        print(f"Crop region: (0, 0) to ({crop_width}, {crop_height})")
        print("=" * 60)
        
        return self.image
    
    
    def crop_dynamic(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """
        Crop a custom rectangular region defined by coordinates.
        
        Parameters:
        -----------
        x1, y1 : int
            Top-left corner coordinates (column, row)
        x2, y2 : int
            Bottom-right corner coordinates (column, row)
            
        Returns:
        --------
        np.ndarray
            Cropped image region
            
        Notes:
        ------
        - x coordinates refer to width (columns)
        - y coordinates refer to height (rows)
        - Array indexing is [row, column], so we use [y, x]
        """
        if self.image is None:
            raise ValueError("No image loaded. Use load_image() first.")
        
        height, width = self.image.shape[:2]
        
        # Clamp coordinates to image boundaries
        # This prevents indexing errors
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        
        # Validate coordinate ordering
        if x1 >= x2 or y1 >= y2:
            raise ValueError(
                "Invalid coordinates: ensure x1 < x2 and y1 < y2"
            )
        
        # Perform the crop
        # Remember: array[row, column] = array[y, x]
        self.image = self.image[y1:y2, x1:x2]
        
        print("\n" + "=" * 60)
        print("DYNAMIC CROP COMPLETED")
        print("=" * 60)
        print(f"Original size: {height}×{width}")
        print(f"Crop coordinates: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"Cropped size: {y2-y1}×{x2-x1}")
        print("=" * 60)
        
        return self.image
    
    
    # ==================== SECTION 4: FLIPPING OPERATIONS ====================
    
    def flip_horizontal(self) -> np.ndarray:
        """
        Flip the image horizontally (mirror effect, left-right reversal).
        
        Returns:
        --------
        np.ndarray
            Horizontally flipped image
        """
        if self.image is None:
            raise ValueError("No image loaded. Use load_image() first.")
        
        # Use NumPy's fliplr (flip left-right) function
        # This reverses the order of columns (axis=1)
        self.image = np.fliplr(self.image)
        
        print("\n" + "=" * 60)
        print("HORIZONTAL FLIP COMPLETED")
        print("=" * 60)
        print("Effect: Left ↔ Right (Mirror)")
        print(f"Image shape: {self.image.shape}")
        print("=" * 60)
        
        return self.image
    
    
    def flip_vertical(self) -> np.ndarray:
        """
        Flip the image vertically (upside down, top-bottom reversal).
        
        Returns:
        --------
        np.ndarray
            Vertically flipped image
        """
        if self.image is None:
            raise ValueError("No image loaded. Use load_image() first.")
        
        # Use NumPy's flipud (flip up-down) function
        # This reverses the order of rows (axis=0)
        self.image = np.flipud(self.image)
        
        print("\n" + "=" * 60)
        print("VERTICAL FLIP COMPLETED")
        print("=" * 60)
        print("Effect: Top ↔ Bottom (Upside Down)")
        print(f"Image shape: {self.image.shape}")
        print("=" * 60)
        
        return self.image
    
    
    def flip_both(self) -> np.ndarray:
        """
        Flip the image both horizontally and vertically (180° rotation).
        
        Returns:
        --------
        np.ndarray
            Image flipped on both axes
        """
        if self.image is None:
            raise ValueError("No image loaded. Use load_image() first.")
        
        # Apply both flips sequentially
        self.image = np.flipud(np.fliplr(self.image))
        
        print("\n" + "=" * 60)
        print("BOTH FLIPS COMPLETED")
        print("=" * 60)
        print("Effect: 180° Rotation")
        print(f"Image shape: {self.image.shape}")
        print("=" * 60)
        
        return self.image
    
    
    # ==================== SECTION 5: VISUALIZATION ====================
    
    def display(self, title: str = "Current Image") -> None:
        """
        Display the current image using matplotlib.
        
        Parameters:
        -----------
        title : str
            Title to display above the image
        """
        if self.image is None:
            raise ValueError("No image loaded. Use load_image() first.")
        
        # Create a figure with specified size
        plt.figure(figsize=(10, 8))
        
        # Display image
        # For grayscale (2D arrays), use gray colormap
        if len(self.image.shape) == 2:
            plt.imshow(self.image, cmap='gray', vmin=0, vmax=255)
        else:
            # For RGB (3D arrays), display normally
            plt.imshow(self.image)
        
        # Add title
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Remove axis ticks for cleaner display
        plt.axis('off')
        
        # Show the plot
        plt.tight_layout()
        plt.show()
    
    
    def display_comparison(self, processed_image: Optional[np.ndarray] = None,
                          original_title: str = "Original",
                          processed_title: str = "Processed") -> None:
        """
        Display original and processed images side by side.
        
        Parameters:
        -----------
        processed_image : np.ndarray, optional
            Processed image to compare. If None, uses current image.
        original_title : str
            Title for the original image
        processed_title : str
            Title for the processed image
        """
        if self.original_image is None:
            raise ValueError("No original image available.")
        
        # Use current image if no processed image provided
        if processed_image is None:
            processed_image = self.image
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display original image
        if len(self.original_image.shape) == 2:
            axes[0].imshow(self.original_image, cmap='gray', vmin=0, vmax=255)
        else:
            axes[0].imshow(self.original_image)
        axes[0].set_title(original_title, fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Display processed image
        if len(processed_image.shape) == 2:
            axes[1].imshow(processed_image, cmap='gray', vmin=0, vmax=255)
        else:
            axes[1].imshow(processed_image)
        axes[1].set_title(processed_title, fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Adjust spacing
        plt.tight_layout()
        plt.show()
    
    
    def save_image(self, output_path: str) -> None:
        """
        Save the current image to a file.
        
        Parameters:
        -----------
        output_path : str
            Path where the image will be saved (e.g., 'output.png')
        """
        if self.image is None:
            raise ValueError("No image to save. Process an image first.")
        
        # Convert NumPy array back to PIL Image
        pil_image = Image.fromarray(self.image)
        
        # Save to file
        # Format is automatically determined from file extension
        pil_image.save(output_path)
        
        print(f"\n✓ Image saved successfully to: {output_path}")
    
    
    def reset(self) -> np.ndarray:
        """
        Reset the image to its original state.
        
        Returns:
        --------
        np.ndarray
            Original image
        """
        if self.original_image is None:
            raise ValueError("No original image available.")
        
        # Restore original image
        self.image = self.original_image.copy()
        
        print("\n✓ Image reset to original state")
        
        return self.image


# ==================== DEMONSTRATION SCRIPT ====================

def create_sample_image():
    """
    Create a colorful test image programmatically.
    This allows testing without needing an external image file.
    
    Returns:
    --------
    np.ndarray
        A sample RGB image (400x600x3)
    """
    print("\n[Creating a sample test image...]")
    
    # Create a blank image (400 height x 600 width x 3 RGB channels)
    height, width = 400, 600
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create colorful gradient sections
    # Red section (top-left quadrant)
    img[0:height//2, 0:width//2, 0] = 255  # Red channel
    
    # Green section (top-right quadrant)
    img[0:height//2, width//2:width, 1] = 255  # Green channel
    
    # Blue section (bottom-left quadrant)
    img[height//2:height, 0:width//2, 2] = 255  # Blue channel
    
    # Yellow section (bottom-right quadrant) - Red + Green
    img[height//2:height, width//2:width, 0] = 255  # Red
    img[height//2:height, width//2:width, 1] = 255  # Green
    
    # Add some circular patterns for visual interest
    center_y, center_x = height // 2, width // 2
    Y, X = np.ogrid[:height, :width]
    
    # Create a white circle in the center
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    circle_mask = dist_from_center <= 80
    img[circle_mask] = [255, 255, 255]  # White
    
    # Create a smaller black circle
    inner_circle = dist_from_center <= 40
    img[inner_circle] = [0, 0, 0]  # Black
    
    print("✓ Sample image created (400×600 pixels)")
    return img


def main():
    """
    Demonstration of all ImageManipulator capabilities.
    This function shows how to use each feature step-by-step.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "IMAGE MANIPULATOR DEMONSTRATION")
    print("=" * 70)
    
    # Initialize the manipulator
    manipulator = ImageManipulator()
    
    # Example 1: Load an image
    print("\n[1] Loading Image...")
    
    # Try to load an existing image file, or create a sample image
    try:
        # OPTION A: Load your own image (uncomment and modify path)
        # manipulator.load_image('path/to/your/image.jpg')
        
        # OPTION B: Create a sample image programmatically (default)
        sample_img = create_sample_image()
        manipulator.image = sample_img
        manipulator.original_image = sample_img.copy()
        print(f"Image shape: {manipulator.image.shape}")
        print(f"Data type: {manipulator.image.dtype}")
        
    except FileNotFoundError:
        print("Please provide a valid image path in the code.")
        return
    
    # Display original image
    manipulator.display("Original Image")
    
    # Example 2: Grayscale conversion
    print("\n[2] Converting to Grayscale...")
    manipulator.convert_to_grayscale()
    manipulator.display_comparison(processed_title="Grayscale")
    
    # Reset to original
    manipulator.reset()
    
    # Example 3: Center crop
    print("\n[3] Applying Center Crop...")
    manipulator.crop_center(crop_height=300, crop_width=300)
    manipulator.display_comparison(processed_title="Center Cropped (300×300)")
    
    # Reset to original
    manipulator.reset()
    
    # Example 4: Top-left crop
    print("\n[4] Applying Top-Left Crop...")
    manipulator.crop_top_left(crop_height=200, crop_width=200)
    manipulator.display("Top-Left Crop (200×200)")
    
    # Reset to original
    manipulator.reset()
    
    # Example 5: Dynamic crop
    print("\n[5] Applying Dynamic Crop...")
    manipulator.crop_dynamic(x1=50, y1=50, x2=400, y2=400)
    manipulator.display("Dynamic Crop (50,50) to (400,400)")
    
    # Reset to original
    manipulator.reset()
    
    # Example 6: Horizontal flip
    print("\n[6] Applying Horizontal Flip...")
    manipulator.flip_horizontal()
    manipulator.display_comparison(processed_title="Horizontally Flipped")
    
    # Reset to original
    manipulator.reset()
    
    # Example 7: Vertical flip
    print("\n[7] Applying Vertical Flip...")
    manipulator.flip_vertical()
    manipulator.display_comparison(processed_title="Vertically Flipped")
    
    # Reset to original
    manipulator.reset()
    
    # Example 8: Combination of operations
    print("\n[8] Applying Multiple Operations...")
    manipulator.convert_to_grayscale()
    manipulator.crop_center(crop_height=400, crop_width=400)
    manipulator.flip_horizontal()
    manipulator.display("Grayscale + Center Crop + Horizontal Flip")
    
    # Save the final result
    manipulator.save_image('output_final.png')
    
    print("\n" + "=" * 70)
    print(" " * 20 + "DEMONSTRATION COMPLETE")
    print("=" * 70)


# Run the demonstration if this file is executed directly
if __name__ == "__main__":
    main()