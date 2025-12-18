
i#Image Compression with SVD: Use SVD to compress an image by keeping only the top k singular values. 
# Show reconstructions with k = 5, 20, 50, 100 and compare file sizes.


"""
PROJECT: IMAGE COMPRESSION WITH SVD
Compress images using Singular Value Decomposition
Compare reconstructions with k = 5, 20, 50, 100 singular values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("="*80)
print("PROJECT: IMAGE COMPRESSION WITH SINGULAR VALUE DECOMPOSITION (SVD)")
print("="*80)

# PART 1: LOAD IMAGE

print("\n" + "="*80)
print("PART 1: LOADING IMAGE")
print("="*80)

# Option 1: Load from URL (sample image)
def load_image_from_url():
    """Load a sample image from the internet"""
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/JPEG_example_flower.jpg/640px-JPEG_example_flower.jpg"
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))
        return np.array(img)
    except:
        print("⚠ Could not download image from URL")
        return None

# Option 2: Load from local file (if you have one)
def load_image_from_file(filepath):
    """Load image from local file"""
    try:
        img = Image.open(filepath)
        return np.array(img)
    except:
        print(f"⚠ Could not load image from {filepath}")
        return None

# Option 3: Create synthetic image for demonstration
def create_synthetic_image(size=(300, 400)):
    """Create a synthetic image with patterns"""
    height, width = size
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create gradient background
    for i in range(height):
        for j in range(width):
            img[i, j, 0] = int(255 * i / height)  # Red gradient
            img[i, j, 1] = int(255 * j / width)   # Green gradient
            img[i, j, 2] = 128                     # Constant blue
    
    # Add some circles
    center_y, center_x = height // 2, width // 2
    Y, X = np.ogrid[:height, :width]
    
    # Circle 1
    mask1 = (X - center_x)**2 + (Y - center_y)**2 <= 50**2
    img[mask1] = [255, 255, 0]  # Yellow
    
    # Circle 2
    mask2 = (X - center_x - 100)**2 + (Y - center_y - 80)**2 <= 30**2
    img[mask2] = [0, 255, 255]  # Cyan
    
    # Circle 3
    mask3 = (X - center_x + 100)**2 + (Y - center_y + 80)**2 <= 40**2
    img[mask3] = [255, 0, 255]  # Magenta
    
    return img

# Try to load image (try URL first, then synthetic)
print("Attempting to load image...")
image = load_image_from_url()

if image is None:
    print("Creating synthetic image for demonstration...")
    image = create_synthetic_image()

print(f"\n✓ Image loaded successfully!")
print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")
print(f"Value range: [{image.min()}, {image.max()}]")

# If grayscale, convert to RGB
if len(image.shape) == 2:
    image = np.stack([image] * 3, axis=-1)
    print("Converted grayscale to RGB")

# PART 2: UNDERSTAND THE IMAGE

print("\n" + "="*80)
print("PART 2: IMAGE ANALYSIS")
print("="*80)

height, width, channels = image.shape
total_pixels = height * width
original_size = image.nbytes

print(f"\nImage dimensions:")
print(f"  Height: {height} pixels")
print(f"  Width: {width} pixels")
print(f"  Channels: {channels} (RGB)")
print(f"  Total pixels: {total_pixels:,}")
print(f"  Original size: {original_size:,} bytes ({original_size/1024:.2f} KB)")

# Display original image
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.imshow(image)
ax.set_title('Original Image', fontsize=14, fontweight='bold')
ax.axis('off')

# Show RGB channels separately
ax = axes[1]
for i, (color, name) in enumerate(zip(['Reds', 'Greens', 'Blues'], ['Red', 'Green', 'Blue'])):
    plt.subplot(1, 3, i+1)
    plt.imshow(image[:, :, i], cmap=color)
    plt.title(f'{name} Channel', fontsize=12, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.savefig('01_original_image.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 01_original_image.png")
plt.show()

# PART 3: SVD COMPRESSION FUNCTION

print("\n" + "="*80)
print("PART 3: SVD COMPRESSION ALGORITHM")
print("="*80)

print("""
SVD COMPRESSION PROCESS:
1. For each color channel (R, G, B):
   - Apply SVD: Channel = U @ Σ @ V^T
   - Keep only top k singular values
   - Reconstruct: Channel_compressed = U[:, :k] @ Σ[:k, :k] @ V^T[:k, :]

2. Compression ratio = Original size / Compressed size
3. Error = ||Original - Compressed|| / ||Original||
""")

def compress_image_svd(image, k):
    """
    Compress image using SVD by keeping top k singular values
    
    Parameters:
    -----------
    image : numpy array
        Input image (height x width x channels)
    k : int
        Number of singular values to keep
    
    Returns:
    --------
    compressed : numpy array
        Compressed image
    components : dict
        Dictionary containing U, sigma, VT for each channel
    stats : dict
        Compression statistics
    """
    height, width, channels = image.shape
    compressed = np.zeros_like(image, dtype=np.float64)
    components = {}
    
    # Original size in bytes
    original_size = image.nbytes
    
    # Compressed size calculation
    # For each channel: U[:, :k] + sigma[:k] + VT[:k, :]
    compressed_size = channels * (height * k + k + k * width) * 8  # 8 bytes per float64
    
    compression_ratio = original_size / compressed_size
    
    for channel in range(channels):
        # Extract channel
        channel_data = image[:, :, channel].astype(np.float64)
        
        # Perform SVD
        U, sigma, VT = np.linalg.svd(channel_data, full_matrices=False)
        
        # Keep only top k singular values
        U_k = U[:, :k]
        sigma_k = sigma[:k]
        VT_k = VT[:k, :]
        
        # Reconstruct
        compressed_channel = U_k @ np.diag(sigma_k) @ VT_k
        
        # Clip values to valid range
        compressed_channel = np.clip(compressed_channel, 0, 255)
        
        compressed[:, :, channel] = compressed_channel
        
        # Store components
        components[channel] = {
            'U': U_k,
            'sigma': sigma_k,
            'VT': VT_k,
            'all_sigma': sigma
        }
    
    # Calculate error
    error = np.linalg.norm(image.astype(np.float64) - compressed) / np.linalg.norm(image.astype(np.float64))
    
    stats = {
        'k': k,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'error': error,
        'percentage_retained': (k / min(height, width)) * 100
    }
    
    return compressed.astype(np.uint8), components, stats

# PART 4: COMPRESS WITH DIFFERENT K VALUES

print("\n" + "="*80)
print("PART 4: COMPRESSING WITH DIFFERENT K VALUES")
print("="*80)

# Test different k values
k_values = [5, 20, 50, 100]
compressed_images = {}
all_stats = []

print(f"\nCompressing image with k = {k_values}...\n")

for k in k_values:
    print(f"Processing k = {k}...")
    compressed_img, components, stats = compress_image_svd(image, k)
    compressed_images[k] = {
        'image': compressed_img,
        'components': components,
        'stats': stats
    }
    all_stats.append(stats)
    
    print(f"  ✓ Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  ✓ Error: {stats['error']:.6f}")
    print(f"  ✓ Size: {stats['compressed_size']:,} bytes ({stats['compressed_size']/1024:.2f} KB)")
    print()

# Create statistics DataFrame
stats_df = pd.DataFrame(all_stats)
print("\nCompression Statistics Summary:")
print(stats_df.to_string(index=False))

# PART 5: VISUALIZE RECONSTRUCTIONS

print("\n" + "="*80)
print("PART 5: VISUALIZING RECONSTRUCTIONS")
print("="*80)

# Create comparison grid
fig, axes = plt.subplots(3, 3, figsize=(16, 14))

# Original image
ax = axes[0, 0]
ax.imshow(image)
ax.set_title(f'Original Image\nSize: {original_size/1024:.1f} KB', 
             fontsize=12, fontweight='bold')
ax.axis('off')

# Compressed images
positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
for (i, j), k in zip(positions, k_values):
    ax = axes[i, j]
    compressed_data = compressed_images[k]
    stats = compressed_data['stats']
    
    ax.imshow(compressed_data['image'])
    ax.set_title(f'k = {k}\n'
                f'Ratio: {stats["compression_ratio"]:.2f}x | '
                f'Error: {stats["error"]:.4f}\n'
                f'Size: {stats["compressed_size"]/1024:.1f} KB',
                fontsize=11, fontweight='bold')
    ax.axis('off')

# Error visualization
ax = axes[1, 2]
for k in k_values:
    original_float = image.astype(np.float64)
    compressed_float = compressed_images[k]['image'].astype(np.float64)
    error_img = np.abs(original_float - compressed_float).mean(axis=2)
    
ax.imshow(error_img, cmap='hot')
ax.set_title(f'Reconstruction Error\n(k={k_values[-1]})', 
             fontsize=12, fontweight='bold')
ax.axis('off')

# Compression ratio chart
ax = axes[2, 0]
ax.bar(range(len(k_values)), stats_df['compression_ratio'], 
       color='steelblue', edgecolor='black', linewidth=2)
ax.set_xticks(range(len(k_values)))
ax.set_xticklabels([f'k={k}' for k in k_values])
ax.set_ylabel('Compression Ratio', fontsize=11, fontweight='bold')
ax.set_title('Compression Ratio Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, ratio in enumerate(stats_df['compression_ratio']):
    ax.text(i, ratio + 0.5, f'{ratio:.2f}x', ha='center', 
            fontweight='bold', fontsize=10)

# Error chart
ax = axes[2, 1]
ax.plot(k_values, stats_df['error'], 'ro-', linewidth=3, markersize=12)
ax.set_xlabel('k (Number of Singular Values)', fontsize=11, fontweight='bold')
ax.set_ylabel('Relative Error', fontsize=11, fontweight='bold')
ax.set_title('Reconstruction Error vs k', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
for k, err in zip(k_values, stats_df['error']):
    ax.text(k, err + 0.01, f'{err:.4f}', ha='center', fontsize=9)

# Size comparison
ax = axes[2, 2]
sizes_kb = [original_size/1024] + list(stats_df['compressed_size']/1024)
labels = ['Original'] + [f'k={k}' for k in k_values]
colors = ['red'] + ['steelblue'] * len(k_values)

bars = ax.bar(range(len(labels)), sizes_kb, color=colors, 
              edgecolor='black', linewidth=2)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Size (KB)', fontsize=11, fontweight='bold')
ax.set_title('File Size Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, (size, label) in enumerate(zip(sizes_kb, labels)):
    ax.text(i, size + max(sizes_kb)*0.02, f'{size:.1f}', 
            ha='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('02_compression_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 02_compression_comparison.png")
plt.show()

# PART 6: SINGULAR VALUE ANALYSIS

print("\n" + "="*80)
print("PART 6: SINGULAR VALUE ANALYSIS")
print("="*80)

# Analyze singular values for each channel
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

channel_names = ['Red', 'Green', 'Blue']
channel_colors = ['red', 'green', 'blue']

# Get singular values from k=100 (has all we need)
components_full = compressed_images[100]['components']

# Plot 1: Singular values for all channels
ax = axes[0, 0]
for ch_idx, (name, color) in enumerate(zip(channel_names, channel_colors)):
    sigma = components_full[ch_idx]['all_sigma']
    ax.plot(sigma[:150], label=f'{name} Channel', 
            color=color, linewidth=2, alpha=0.7)

for k in k_values:
    ax.axvline(k, color='gray', linestyle='--', alpha=0.5)
    ax.text(k, ax.get_ylim()[1]*0.9, f'k={k}', 
            rotation=90, ha='right', fontsize=9)

ax.set_xlabel('Index', fontsize=11, fontweight='bold')
ax.set_ylabel('Singular Value', fontsize=11, fontweight='bold')
ax.set_title('Singular Values (All Channels)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Log scale
ax = axes[0, 1]
for ch_idx, (name, color) in enumerate(zip(channel_names, channel_colors)):
    sigma = components_full[ch_idx]['all_sigma']
    ax.semilogy(sigma[:150], label=f'{name} Channel', 
                color=color, linewidth=2, alpha=0.7)

for k in k_values:
    ax.axvline(k, color='gray', linestyle='--', alpha=0.5)

ax.set_xlabel('Index', fontsize=11, fontweight='bold')
ax.set_ylabel('Singular Value (log scale)', fontsize=11, fontweight='bold')
ax.set_title('Singular Values (Log Scale)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# Plot 3: Cumulative energy
ax = axes[1, 0]
for ch_idx, (name, color) in enumerate(zip(channel_names, channel_colors)):
    sigma = components_full[ch_idx]['all_sigma']
    cumulative_energy = np.cumsum(sigma**2) / np.sum(sigma**2)
    ax.plot(cumulative_energy[:150], label=f'{name} Channel', 
            color=color, linewidth=2, alpha=0.7)

for k in k_values:
    ax.axvline(k, color='gray', linestyle='--', alpha=0.5)
    # Show energy at this k
    sigma = components_full[0]['all_sigma']
    energy = np.sum(sigma[:k]**2) / np.sum(sigma**2)
    ax.text(k, energy - 0.05, f'{energy:.1%}', 
            ha='center', fontsize=8, bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.7))

ax.axhline(0.9, color='red', linestyle=':', linewidth=2, label='90% Energy')
ax.set_xlabel('Number of Components (k)', fontsize=11, fontweight='bold')
ax.set_ylabel('Cumulative Energy', fontsize=11, fontweight='bold')
ax.set_title('Energy Preserved by k Components', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Energy retained for each k
ax = axes[1, 1]
energy_data = []
for k in k_values:
    energies = []
    for ch_idx in range(3):
        sigma = components_full[ch_idx]['all_sigma']
        energy = np.sum(sigma[:k]**2) / np.sum(sigma**2)
        energies.append(energy)
    energy_data.append(energies)

energy_array = np.array(energy_data).T
x = np.arange(len(k_values))
width = 0.25

for i, (name, color) in enumerate(zip(channel_names, channel_colors)):
    ax.bar(x + i*width, energy_array[i], width, 
           label=f'{name} Channel', color=color, alpha=0.7,
           edgecolor='black', linewidth=1.5)

ax.set_xlabel('k Value', fontsize=11, fontweight='bold')
ax.set_ylabel('Energy Retained', fontsize=11, fontweight='bold')
ax.set_title('Energy Retained by Channel', fontsize=12, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([f'k={k}' for k in k_values])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('03_singular_value_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 03_singular_value_analysis.png")
plt.show()

# PART 7: DETAILED COMPARISON TABLE

print("\n" + "="*80)
print("PART 7: DETAILED COMPARISON")
print("="*80)

# Create detailed comparison
comparison_data = []

# Add original
comparison_data.append({
    'Configuration': 'Original',
    'k': '-',
    'Size (KB)': original_size / 1024,
    'Compression Ratio': 1.0,
    'Error': 0.0,
    'PSNR (dB)': np.inf,
    'Quality': '100%'
})

# Add compressed versions
for k in k_values:
    stats = compressed_images[k]['stats']
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    mse = np.mean((image.astype(np.float64) - 
                   compressed_images[k]['image'].astype(np.float64))**2)
    if mse == 0:
        psnr = np.inf
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    quality = (1 - stats['error']) * 100
    
    comparison_data.append({
        'Configuration': f'k = {k}',
        'k': k,
        'Size (KB)': stats['compressed_size'] / 1024,
        'Compression Ratio': stats['compression_ratio'],
        'Error': stats['error'],
        'PSNR (dB)': psnr,
        'Quality': f'{quality:.2f}%'
    })

comparison_df = pd.DataFrame(comparison_data)

print("\nDetailed Comparison Table:")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)

# PART 8: VISUAL QUALITY ASSESSMENT

print("\n" + "="*80)
print("PART 8: VISUAL QUALITY ASSESSMENT")
print("="*80)

# Zoom in on a specific region to see detail
# Take center crop
crop_size = 100
cy, cx = height // 2, width // 2
y1, y2 = cy - crop_size//2, cy + crop_size//2
x1, x2 = cx - crop_size//2, cx + crop_size//2

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Original crop
ax = axes[0, 0]
ax.imshow(image[y1:y2, x1:x2])
ax.set_title('Original (Detail)', fontsize=12, fontweight='bold')
ax.axis('off')

# Compressed crops
positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
for (i, j), k in zip(positions[:4], k_values):
    ax = axes[i, j]
    crop = compressed_images[k]['image'][y1:y2, x1:x2]
    ax.imshow(crop)
    ax.set_title(f'k = {k} (Detail)', fontsize=12, fontweight='bold')
    ax.axis('off')

# Difference map for k=5
ax = axes[1, 2]
diff = np.abs(image[y1:y2, x1:x2].astype(np.float64) - 
              compressed_images[5]['image'][y1:y2, x1:x2].astype(np.float64))
diff_map = diff.mean(axis=2)
im = ax.imshow(diff_map, cmap='hot')
ax.set_title('Error Map (k=5)', fontsize=12, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('04_quality_assessment.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 04_quality_assessment.png")
plt.show()

# PART 9: FINAL SUMMARY

print("\n" + "="*80)
print("FINAL PROJECT SUMMARY")
print("="*80)

summary = f"""
IMAGE COMPRESSION WITH SVD - COMPLETE ANALYSIS

ORIGINAL IMAGE:
- Dimensions: {height} × {width} pixels
- Channels: {channels} (RGB)
- Size: {original_size:,} bytes ({original_size/1024:.2f} KB)

COMPRESSION RESULTS:

k = 5:
  ✓ Compression Ratio: {compressed_images[5]['stats']['compression_ratio']:.2f}x
  ✓ Size: {compressed_images[5]['stats']['compressed_size']/1024:.2f} KB
  ✓ Error: {compressed_images[5]['stats']['error']:.4f}
  ✓ Quality: Very low (suitable for thumbnails)

k = 20:
  ✓ Compression Ratio: {compressed_images[20]['stats']['compression_ratio']:.2f}x
  ✓ Size: {compressed_images[20]['stats']['compressed_size']/1024:.2f} KB
  ✓ Error: {compressed_images[20]['stats']['error']:.4f}
  ✓ Quality: Low-medium (recognizable but blurry)

k = 50:
  ✓ Compression Ratio: {compressed_images[50]['stats']['compression_ratio']:.2f}x
  ✓ Size: {compressed_images[50]['stats']['compressed_size']/1024:.2f} KB
  ✓ Error: {compressed_images[50]['stats']['error']:.4f}
  ✓ Quality: Good (suitable for most applications)

k = 100:
  ✓ Compression Ratio: {compressed_images[100]['stats']['compression_ratio']:.2f}x
  ✓ Size: {compressed_images[100]['stats']['compressed_size']/1024:.2f} KB
  ✓ Error: {compressed_images[100]['stats']['error']:.4f}
  ✓ Quality: Very good (hard to distinguish from original)

KEY INSIGHTS:
1. SVD enables progressive quality compression
2. Larger k values preserve more detail but reduce compression
3. Trade-off between quality and file size
4. First few singular values capture most image energy
5. Works independently on each color channel

FILES GENERATED:
- 01_original_image.png (original + channels)
- 02_compression_comparison.png (reconstructions + stats)
- 03_singular_value_analysis.png (singular value plots)
- 04_quality_assessment.png (detailed quality view)

TOTAL: 4 visualization files

RECOMMENDATIONS:
- k = 5-10: Use for extreme compression (previews)
- k = 20-50: Use for web images (good quality/size balance)
- k = 100+: Use when quality is critical

PROJECT COMPLETE! 🎉
"""

print(summary)

# Save summary to file
with open('compression_summary.txt', 'w') as f:
    f.write(summary)
print("\n✓ Saved: compression_summary.txt")

print("\n" + "="*80)
print("Thank you for running the SVD Image Compression project!")
print("="*80)