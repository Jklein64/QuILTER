import numpy as np
from PIL import Image
import os

from time import perf_counter

# determined through experimentation
GUIDED_FILTER_RADIUS = 30
# taken from example on GitHub
GUIDED_FILTER_EPSILON = 1e-6 * 255 ** 2


def main():
    # loads everything as 8-bit, 3-channel images
    image = np.array(Image.open("horse/horse.png"))[..., 0:3]
    masks = [np.array(Image.open(f"horse/masks/{file}")) for file in os.listdir("horse/masks")]

    start = perf_counter()

    filtered = filter(np.stack(masks, axis=-1), image).astype(int)

    # scale to keep the ratio between masks but have a max sum of 255
    scaled = (255 * filtered / np.sum(filtered, axis=-1, keepdims=True)).astype(np.uint8)
    ascending_i = np.argsort(scaled, axis=-1)
    union = np.sum(scaled, axis=-1)
    # absolute value to keep as uint8; direction is from union > 255
    difference = np.absolute(union.astype(int) - 255).astype(np.uint8)

    def add_difference(n, difference):
        """Add the given difference to the n'th largest channel.  If a value in the channel would integer overflow (for an 8-bit integer), then it it set to the maximum value instead.  Call this function but with an increased value of n and an updated difference array until there is no longer anything to add.  Solves the note above."""
        # get the index for the n'th largest channel
        channel = len(masks) - n - 1
        can_add_whole = 255 - scaled[union < 255, ascending_i[union < 255, ..., channel]] >= difference[union < 255]
        # add the entire difference to values that can take it without overflowing
        selection = (ascending_i[union < 255] == n) & can_add_whole[:, np.newaxis]  
        scaled[union < 255] += np.where(selection, difference[union < 255, np.newaxis], 0)
        # add as much as possible to ones that would overflow; equivalently, set them to the max
        scaled[union < 255][~can_add_whole] = 255

    i = 0
    while not np.all(difference == 0):
        add_difference(i, difference)
        union = np.sum(scaled, axis=-1)
        difference = np.absolute(union.astype(int) - 255).astype(np.uint8)
        i += 1

    end = perf_counter()
    print(f"took {end - start} seconds to complete the operation.")

    if i == 1:
        print("finished on the first iteration.  Do you really need the repetitive process?")

    pass


def show(image):
    Image.fromarray(image).show()


def pca(data, dim):
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


def filter(image, guide):
    from cv2.ximgproc import guidedFilter
    return guidedFilter(guide, image, GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPSILON)


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