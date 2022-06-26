from argparse import ArgumentParser
from PIL import Image

import numpy as np
import os


def main():
    parser = ArgumentParser(description="Feather binary masks from semantic segmentation using a guided filter without affecting their union.")
    parser.add_argument("image", help="Path to the original image file")
    parser.add_argument("masks", help="Path to a directory containing binary masks for each segment; the union of the masks covers the whole image")
    parser.add_argument("-r", "--radius", type=int, default=8, help="Guided filter window radius, default is 8")
    parser.add_argument("-e", "--epsilon", type=float, default=1e-6, help="Guided filter epsilon, default is 1e-6")
    parser.add_argument("-o", "--output", help="Output directory for the feathered masks. Not supplying this argument will show them instead")
    args = parser.parse_args()

    # loads everything as 8-bit, 3-channel images
    image = np.array(Image.open(args.image))[..., 0:3]
    masks = [np.array(Image.open(os.path.join(args.masks, file))) for file in os.listdir(args.masks)]

    # store masks as if they were per-pixel features
    filtered = filter(np.stack(masks, axis=-1), image, args.radius, args.epsilon).astype(int)
    # scale to keep the ratio between mask opacities but have a max sum of 255
    scaled = (255 * filtered / np.sum(filtered, axis=-1, keepdims=True)).astype(np.uint8)
    union = np.sum(scaled, axis=-1)
    # absolute value to keep as uint8; direction is from union > 255
    difference = np.absolute(union.astype(int) - 255).astype(np.uint8)
    # expand dims to add the difference to the largest value
    ascending_indices = np.argsort(scaled, axis=-1)
    largest_value = ascending_indices[union < 255] == len(masks) - 1
    difference = np.where(largest_value, difference[union < 255, np.newaxis], 0)
    # np.all(np.sum(scaled, axis=-1) == 255) should be True after this addition
    scaled[union < 255] += difference

    for i in range(len(masks)):
        output = Image.fromarray(scaled[..., i])
        if args.output:
            output.save(args.output)
        else:
            output.show()


def show(image):
    """Helper function for debugging"""
    Image.fromarray(image).show()


def pca(data, dim):
    # should this be standardized?
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    standardized = (data - mu) / sigma
    covariance = np.cov(standardized.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    order = np.flip(np.argsort(eigenvalues))
    basis = eigenvectors[..., order][..., :dim]
    projected = data @ basis
    projected -= np.min(projected, axis=0)
    projected /= np.max(projected, axis=0)
    return projected


def filter(image, guide, radius, epsilon):
    from cv2.ximgproc import guidedFilter
    return guidedFilter(guide, image, radius, epsilon)


def apply_mask(image, mask):
    """Use mask as alpha channel in image."""
    return np.insert(image[..., 0:3], 3, mask, axis=-1)


def apply_pca(image_like):
    """Given per-pixel data, project to 3 dimensions with PCA and visualize."""
    w, h, d = np.shape(image_like)
    data = np.reshape(image_like, (w * h, d))
    projected = np.reshape(pca(data, dim=3), (w, h, 3))
    return (projected * 255).astype(np.uint8)


if __name__ == "__main__":
    main()