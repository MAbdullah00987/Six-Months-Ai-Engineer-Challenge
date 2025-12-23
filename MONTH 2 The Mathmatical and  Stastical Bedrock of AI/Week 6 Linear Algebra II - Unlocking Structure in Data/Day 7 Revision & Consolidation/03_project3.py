
#3: Image Compression with SVD: Use SVD to compress an image by keeping only the top 'k' singular values and
# reconstructing the image.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO

# Set style
sns.set_style("white")
plt.rcParams['figure.figsize'] = (18, 12)

def load_sample_image():
    """Load a sample image from the internet"""
    # Using a sample image URL (a nature photo)
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/800px-Fronalpstock_big.jpg"
    
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        # Resize for faster processing
        img = img.resize((400, 300))
        return np.array(img)
    except:
        # Create a synthetic image if download fails
        print("Creating synthetic image...")
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        # Create a gradient pattern
        for i in range(300):
            for j in range(400):
                img[i, j] = [i % 256, j % 256, (i+j) % 256]
        return img

def compress_channel_svd(channel, k):
    """
    Compress a single color channel using SVD
    
    Parameters:
    - channel: 2D numpy array (single color channel)
    - k: number of singular values to keep
    
    Returns:
    - compressed channel
    - U, S, Vt matrices
    """
    # Perform SVD: A = U @ S @ Vt
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    
    # Keep only top k singular values
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    # Reconstruct the channel
    compressed = U_k @ np.diag(S_k) @ Vt_k
    
    # Clip values to valid range [0, 255]
    compressed = np.clip(compressed, 0, 255)
    
    return compressed, U, S, Vt

def compress_image_svd(image, k):
    """
    Compress a color image using SVD on each channel
    
    Parameters:
    - image: RGB image (H x W x 3)
    - k: number of singular values to keep per channel
    
    Returns:
    - compressed image
    - singular values for each channel
    """
    if len(image.shape) == 2:
        # Grayscale image
        compressed, U, S, Vt = compress_channel_svd(image, k)
        return compressed.astype(np.uint8), [S]
    
    # RGB image - compress each channel
    compressed_channels = []
    all_singular_values = []
    
    for i in range(3):  # R, G, B channels
        channel = image[:, :, i].astype(float)
        compressed_channel, U, S, Vt = compress_channel_svd(channel, k)
        compressed_channels.append(compressed_channel)
        all_singular_values.append(S)
    
    # Stack channels back together
    compressed_image = np.stack(compressed_channels, axis=2)
    
    return compressed_image.astype(np.uint8), all_singular_values

def calculate_compression_ratio(original_shape, k, n_channels=3):
    """Calculate compression ratio"""
    m, n = original_shape[:2]
    
    # Original size
    original_size = m * n * n_channels
    
    # Compressed size: (U_k + S_k + Vt_k) per channel
    # U_k: m x k, S_k: k, Vt_k: k x n
    compressed_size = n_channels * (m * k + k + k * n)
    
    ratio = original_size / compressed_size
    return ratio

def calculate_mse(original, compressed):
    """Calculate Mean Squared Error"""
    return np.mean((original.astype(float) - compressed.astype(float)) ** 2)

def calculate_psnr(original, compressed):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = calculate_mse(original, compressed)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# ============================================================================
# Load and Display Original Image
# ============================================================================
print("="*70)
print("LOADING IMAGE")
print("="*70)

original_image = load_sample_image()
print(f"Image shape: {original_image.shape}")
print(f"Image dtype: {original_image.dtype}")
print(f"Value range: [{original_image.min()}, {original_image.max()}]")
print()

# ============================================================================
# Compress with Different k values
# ============================================================================
k_values = [5, 10, 20, 50, 100]
compressed_images = {}
metrics = {}

print("="*70)
print("COMPRESSING IMAGE WITH DIFFERENT k VALUES")
print("="*70)

for k in k_values:
    print(f"\nCompressing with k = {k}...")
    compressed, singular_values = compress_image_svd(original_image, k)
    compressed_images[k] = compressed
    
    # Calculate metrics
    compression_ratio = calculate_compression_ratio(original_image.shape, k)
    mse = calculate_mse(original_image, compressed)
    psnr = calculate_psnr(original_image, compressed)
    
    metrics[k] = {
        'compression_ratio': compression_ratio,
        'mse': mse,
        'psnr': psnr,
        'singular_values': singular_values
    }
    
    print(f"  Compression Ratio: {compression_ratio:.2f}x")
    print(f"  MSE: {mse:.2f}")
    print(f"  PSNR: {psnr:.2f} dB")

# ============================================================================
# Visualizations
# ============================================================================

fig = plt.figure(figsize=(20, 14))

# Plot 1: Original Image
plt.subplot(3, 4, 1)
plt.imshow(original_image)
plt.title(f'Original Image\n{original_image.shape[0]}x{original_image.shape[1]}', 
          fontsize=12, fontweight='bold')
plt.axis('off')

# Plot 2-6: Compressed Images
for idx, k in enumerate(k_values, start=2):
    plt.subplot(3, 4, idx)
    plt.imshow(compressed_images[k])
    ratio = metrics[k]['compression_ratio']
    psnr = metrics[k]['psnr']
    plt.title(f'k = {k}\nRatio: {ratio:.1f}x | PSNR: {psnr:.1f} dB', 
              fontsize=11)
    plt.axis('off')

# Plot 7: Compression Ratio vs k
plt.subplot(3, 4, 7)
ratios = [metrics[k]['compression_ratio'] for k in k_values]
plt.plot(k_values, ratios, 'bo-', linewidth=2, markersize=8)
plt.xlabel('k (number of singular values)', fontsize=11)
plt.ylabel('Compression Ratio', fontsize=11)
plt.title('Compression Ratio vs k', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
for i, k in enumerate(k_values):
    plt.text(k, ratios[i], f'{ratios[i]:.1f}x', 
             ha='center', va='bottom', fontsize=9)

# Plot 8: PSNR vs k
plt.subplot(3, 4, 8)
psnr_values = [metrics[k]['psnr'] for k in k_values]
plt.plot(k_values, psnr_values, 'ro-', linewidth=2, markersize=8)
plt.xlabel('k (number of singular values)', fontsize=11)
plt.ylabel('PSNR (dB)', fontsize=11)
plt.title('Image Quality (PSNR) vs k', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Good quality')
plt.legend()

# Plot 9: MSE vs k
plt.subplot(3, 4, 9)
mse_values = [metrics[k]['mse'] for k in k_values]
plt.plot(k_values, mse_values, 'go-', linewidth=2, markersize=8)
plt.xlabel('k (number of singular values)', fontsize=11)
plt.ylabel('Mean Squared Error', fontsize=11)
plt.title('Reconstruction Error (MSE) vs k', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 10: Singular Values Distribution (Red channel)
plt.subplot(3, 4, 10)
# Get singular values for k=100 (or max k)
max_k = max(k_values)
S_red = metrics[max_k]['singular_values'][0]
S_green = metrics[max_k]['singular_values'][1]
S_blue = metrics[max_k]['singular_values'][2]

plt.plot(S_red[:100], 'r-', label='Red', linewidth=2, alpha=0.7)
plt.plot(S_green[:100], 'g-', label='Green', linewidth=2, alpha=0.7)
plt.plot(S_blue[:100], 'b-', label='Blue', linewidth=2, alpha=0.7)
plt.xlabel('Singular Value Index', fontsize=11)
plt.ylabel('Singular Value Magnitude', fontsize=11)
plt.title('Singular Values by Channel', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 11: Cumulative Energy
plt.subplot(3, 4, 11)
cumsum_red = np.cumsum(S_red**2) / np.sum(S_red**2)
cumsum_green = np.cumsum(S_green**2) / np.sum(S_green**2)
cumsum_blue = np.cumsum(S_blue**2) / np.sum(S_blue**2)

plt.plot(cumsum_red[:150], 'r-', label='Red', linewidth=2, alpha=0.7)
plt.plot(cumsum_green[:150], 'g-', label='Green', linewidth=2, alpha=0.7)
plt.plot(cumsum_blue[:150], 'b-', label='Blue', linewidth=2, alpha=0.7)
plt.axhline(y=0.9, color='k', linestyle='--', alpha=0.5, label='90% energy')
plt.axhline(y=0.95, color='k', linestyle=':', alpha=0.5, label='95% energy')
plt.xlabel('Number of Components', fontsize=11)
plt.ylabel('Cumulative Energy Ratio', fontsize=11)
plt.title('Cumulative Energy vs Components', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 12: Quality vs Compression Trade-off
plt.subplot(3, 4, 12)
plt.scatter(ratios, psnr_values, s=200, c=k_values, cmap='viridis', 
            edgecolors='black', linewidth=2)
plt.xlabel('Compression Ratio', fontsize=11)
plt.ylabel('PSNR (dB)', fontsize=11)
plt.title('Quality vs Compression Trade-off', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
cbar = plt.colorbar(label='k value')

# Add annotations
for i, k in enumerate(k_values):
    plt.annotate(f'k={k}', (ratios[i], psnr_values[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.tight_layout()
plt.show()

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("COMPRESSION SUMMARY")
print("="*70)

summary_df = []
for k in k_values:
    summary_df.append({
        'k': k,
        'Compression Ratio': f"{metrics[k]['compression_ratio']:.2f}x",
        'MSE': f"{metrics[k]['mse']:.2f}",
        'PSNR (dB)': f"{metrics[k]['psnr']:.2f}",
        'Quality': 'Excellent' if metrics[k]['psnr'] > 35 else 
                   'Good' if metrics[k]['psnr'] > 30 else 
                   'Fair' if metrics[k]['psnr'] > 25 else 'Poor'
    })

import pandas as pd
df = pd.DataFrame(summary_df)
print(df.to_string(index=False))

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print("• SVD decomposes image matrix: A = U @ S @ Vt")
print("• Keeping top k singular values compresses while preserving quality")
print("• Larger k = better quality but less compression")
print("• PSNR > 30 dB is generally considered good quality")
print("• Red channel typically has most energy (natural images)")
print("="*70)