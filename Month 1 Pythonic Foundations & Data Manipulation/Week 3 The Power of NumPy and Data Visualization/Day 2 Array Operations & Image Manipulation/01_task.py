
import numpy as np

class ImageManipulator:
    """Image manipulation using NumPy arrays only"""
    
    def __init__(self, width=100, height=100, channels=3):
        """Initialize with a synthetic image"""
        self.width = width
        self.height = height
        self.channels = channels
        self.image = self.create_sample_image()
    
    def create_sample_image(self):
        """Create a sample RGB image with gradient patterns"""
        img = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)
        
        # Create gradient patterns for each channel
        for i in range(self.height):
            for j in range(self.width):
                img[i, j, 0] = int(255 * i / self.height)  # Red gradient (vertical)
                img[i, j, 1] = int(255 * j / self.width)   # Green gradient (horizontal)
                img[i, j, 2] = int(255 * (i + j) / (self.height + self.width))  # Blue diagonal
        
        return img
    
    def get_image_info(self):
        """Display image information"""
        print("=" * 50)
        print("IMAGE INFORMATION")
        print("=" * 50)
        print(f"Shape: {self.image.shape}")
        print(f"Size: {self.image.size} pixels")
        print(f"Data type: {self.image.dtype}")
        print(f"Min value: {self.image.min()}")
        print(f"Max value: {self.image.max()}")
        print(f"Mean value: {self.image.mean():.2f}")
        print()
    
    def crop(self, x1, y1, x2, y2):
        """Crop image to specified region"""
        cropped = self.image[y1:y2, x1:x2]
        print(f"Cropped from {self.image.shape} to {cropped.shape}")
        return cropped
    
    def flip_horizontal(self):
        """Flip image horizontally (mirror)"""
        flipped = self.image[:, ::-1, :]
        print("Flipped horizontally")
        return flipped
    
    def flip_vertical(self):
        """Flip image vertically"""
        flipped = self.image[::-1, :, :]
        print("Flipped vertically")
        return flipped
    
    def rotate_90(self):
        """Rotate image 90 degrees clockwise"""
        # Transpose and flip for 90-degree rotation
        rotated = np.transpose(self.image, (1, 0, 2))[::-1, :, :]
        print(f"Rotated 90° clockwise: {self.image.shape} -> {rotated.shape}")
        return rotated
    
    def rotate_180(self):
        """Rotate image 180 degrees"""
        rotated = self.image[::-1, ::-1, :]
        print("Rotated 180°")
        return rotated
    
    def grayscale(self):
        """Convert to grayscale using luminosity method"""
        # Weighted average: 0.299*R + 0.587*G + 0.114*B
        weights = np.array([0.299, 0.587, 0.114])
        gray = np.dot(self.image, weights).astype(np.uint8)
        print(f"Converted to grayscale: {self.image.shape} -> {gray.shape}")
        return gray
    
    def brightness(self, factor):
        """Adjust brightness (factor: 0.5=darker, 1.5=brighter)"""
        adjusted = np.clip(self.image * factor, 0, 255).astype(np.uint8)
        print(f"Brightness adjusted by factor {factor}")
        return adjusted
    
    def contrast(self, factor):
        """Adjust contrast (factor: 0.5=less, 1.5=more)"""
        mean = self.image.mean()
        adjusted = np.clip((self.image - mean) * factor + mean, 0, 255).astype(np.uint8)
        print(f"Contrast adjusted by factor {factor}")
        return adjusted
    
    def threshold(self, value=128):
        """Apply binary threshold (black and white)"""
        gray = self.grayscale()
        binary = np.where(gray > value, 255, 0).astype(np.uint8)
        print(f"Applied threshold at {value}")
        return binary
    
    def invert(self):
        """Invert colors (negative)"""
        inverted = 255 - self.image
        print("Colors inverted")
        return inverted
    
    def extract_channel(self, channel):
        """Extract single color channel (0=R, 1=G, 2=B)"""
        channel_names = {0: 'Red', 1: 'Green', 2: 'Blue'}
        extracted = self.image[:, :, channel]
        print(f"Extracted {channel_names[channel]} channel")
        return extracted
    
    def swap_channels(self, order):
        """Swap color channels (e.g., [2, 1, 0] for BGR)"""
        swapped = self.image[:, :, order]
        print(f"Channels swapped to order {order}")
        return swapped
    
    def resize(self, new_height, new_width):
        """Resize image using nearest neighbor (simple method)"""
        # Calculate step sizes
        row_step = self.height / new_height
        col_step = self.width / new_width
        
        # Create index arrays
        row_indices = (np.arange(new_height) * row_step).astype(int)
        col_indices = (np.arange(new_width) * col_step).astype(int)
        
        # Clip to valid range
        row_indices = np.clip(row_indices, 0, self.height - 1)
        col_indices = np.clip(col_indices, 0, self.width - 1)
        
        # Use fancy indexing to resize
        resized = self.image[row_indices[:, np.newaxis], col_indices, :]
        print(f"Resized from {self.image.shape[:2]} to {resized.shape[:2]}")
        return resized
    
    def add_border(self, thickness=5, color=[255, 0, 0]):
        """Add colored border around image"""
        bordered = self.image.copy()
        bordered[:thickness, :] = color  # Top
        bordered[-thickness:, :] = color  # Bottom
        bordered[:, :thickness] = color  # Left
        bordered[:, -thickness:] = color  # Right
        print(f"Added {thickness}px border")
        return bordered
    
    def create_mosaic(self, rows=2, cols=2):
        """Create mosaic by repeating image"""
        mosaic = np.tile(self.image, (rows, cols, 1))
        print(f"Created {rows}x{cols} mosaic: {mosaic.shape}")
        return mosaic
    
    def apply_mask(self, condition):
        """Apply boolean mask based on condition"""
        # Example: Keep only bright pixels
        if condition == 'bright':
            mask = self.image.mean(axis=2) > 128
        elif condition == 'dark':
            mask = self.image.mean(axis=2) <= 128
        elif condition == 'red':
            mask = self.image[:, :, 0] > self.image[:, :, 1:].mean(axis=2)
        else:
            mask = np.ones((self.height, self.width), dtype=bool)
        
        # Apply mask
        masked = self.image.copy()
        masked[~mask] = 0
        print(f"Applied '{condition}' mask, kept {mask.sum()} pixels")
        return masked
    
    def get_statistics(self):
        """Calculate and display image statistics"""
        print("=" * 50)
        print("IMAGE STATISTICS")
        print("=" * 50)
        print(f"Overall - Mean: {self.image.mean():.2f}, Std: {self.image.std():.2f}")
        print(f"Red   - Mean: {self.image[:,:,0].mean():.2f}, Std: {self.image[:,:,0].std():.2f}")
        print(f"Green - Mean: {self.image[:,:,1].mean():.2f}, Std: {self.image[:,:,1].std():.2f}")
        print(f"Blue  - Mean: {self.image[:,:,2].mean():.2f}, Std: {self.image[:,:,2].std():.2f}")
        print()


# ============================================
# DEMONSTRATION
# ============================================

def main():
    print("\n" + "="*60)
    print("IMAGE MANIPULATOR - NumPy Edition")
    print("="*60 + "\n")
    
    # Create image manipulator
    img_manip = ImageManipulator(width=80, height=60)
    img_manip.get_image_info()
    img_manip.get_statistics()
    
    # Demonstrate transformations
    print("TRANSFORMATIONS")
    print("-" * 50)
    
    # 1. Cropping
    cropped = img_manip.crop(10, 10, 50, 40)
    
    # 2. Flipping
    h_flipped = img_manip.flip_horizontal()
    v_flipped = img_manip.flip_vertical()
    
    # 3. Rotation
    rotated_90 = img_manip.rotate_90()
    rotated_180 = img_manip.rotate_180()
    
    # 4. Grayscale conversion
    gray = img_manip.grayscale()
    
    # 5. Brightness and contrast
    brighter = img_manip.brightness(1.5)
    darker = img_manip.brightness(0.5)
    high_contrast = img_manip.contrast(1.5)
    
    # 6. Threshold
    binary = img_manip.threshold(128)
    
    # 7. Color inversion
    inverted = img_manip.invert()
    
    # 8. Channel operations
    red_channel = img_manip.extract_channel(0)
    bgr_image = img_manip.swap_channels([2, 1, 0])
    
    # 9. Resizing
    small = img_manip.resize(30, 40)
    large = img_manip.resize(120, 160)
    
    # 10. Border
    bordered = img_manip.add_border(thickness=5, color=[255, 255, 0])
    
    # 11. Mosaic
    mosaic = img_manip.create_mosaic(2, 3)
    
    # 12. Masking
    bright_pixels = img_manip.apply_mask('bright')
    red_pixels = img_manip.apply_mask('red')
    
    print("\n" + "="*60)
    print("ADVANCED ARRAY OPERATIONS EXAMPLES")
    print("="*60 + "\n")
    
    # Stacking examples
    print("STACKING OPERATIONS:")
    print("-" * 50)
    img1 = img_manip.crop(0, 0, 40, 30)
    img2 = img_manip.crop(40, 30, 80, 60)
    
    vstacked = np.vstack((img1, img2))
    print(f"Vertical stack: {img1.shape} + {img2.shape} = {vstacked.shape}")
    
    hstacked = np.hstack((img1, img2))
    print(f"Horizontal stack: {img1.shape} + {img2.shape} = {hstacked.shape}")
    
    # Fancy indexing example
    print("\nFANCY INDEXING:")
    print("-" * 50)
    rows = [0, 10, 20, 30]
    cols = [0, 20, 40, 60]
    sampled = img_manip.image[np.ix_(rows, cols)]
    print(f"Sampled {len(rows)}x{len(cols)} pixels from original image")
    print(f"Result shape: {sampled.shape}")
    
    # Boolean indexing example
    print("\nBOOLEAN INDEXING:")
    print("-" * 50)
    bright_mask = img_manip.image.mean(axis=2) > 150
    print(f"Pixels with average brightness > 150: {bright_mask.sum()}/{bright_mask.size}")
    print(f"Percentage: {100*bright_mask.sum()/bright_mask.size:.1f}%")
    
    print("\n" + "="*60)
    print("All operations completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()