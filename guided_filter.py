import numpy as np
from PIL import Image

# determined through experimentation
GUIDED_FILTER_RADIUS = 30
# taken from example on GitHub
GUIDED_FILTER_EPSILON = 1e-6 * 255 ** 2

def main():
    # loads everything as 8-bit images
    image = np.array(Image.open("original.png"))
    masks = [np.array(Image.open(f"{i}.png")) for i in (1, 2, 3)]
    filtered = filter(np.stack(masks, axis=-1), image).astype(int)

    union = np.sum(filtered, axis=-1)
    # prefer sparsity
    # subtract from smallest channel
    # add to largest channel

    # absolute value to keep as uint8; direction is from union > 255
    difference = np.absolute(union.astype(int) - 255).astype(np.uint8)

    # handle where the sum of the masks is too large
    # subtract from smallest channels to increase sparsity
    smallest_i = np.argsort(filtered[union > 255], axis=1)
    # initially subtract from the smallest channel, allowing for negatives
    filtered[union > 255, smallest_i[..., 0]] -= difference[union > 255]
    # cancel negatives by adding to the next largest channel and setting to zero
    ordered = np.take_along_axis(filtered[union > 255], smallest_i, axis=-1)
    for channel in range(len(masks) - 1):
        ordered_negative = ordered[np.where(ordered < 0)[0]]
        ordered_negative[..., channel + 1] += ordered_negative[..., channel]
        ordered_negative[..., channel] = 0
    # TODO extend for multiple dimensions

    # filtered[union > 255, smallest] -= difference[union > 255]

    # handle where the sum of the masks is too small
    # add to largest channel to increase sparsity
    largest_i = np.argmax(filtered[union < 255], axis=-1)
    filtered[union < 255, largest_i] += difference[union < 255]

    pass


def show(image):
    Image.fromarray(image).show()


def filter(image, guide):
    from cv2.ximgproc import guidedFilter
    return guidedFilter(guide, image, GUIDED_FILTER_RADIUS, GUIDED_FILTER_EPSILON)


def apply_mask(image, mask):
    return np.insert(image[..., 0:3], 3, mask, axis=-1)


if __name__ == "__main__":
    main()

# it turns out that part of the issue is also integer overflow.
# in addition to having parts of the image that weren't fully
# covered by the union of the masks, there were others that 
# were over-covered by the union.  Integer overflow made it
# look like these areas weren't covered at all :O

# where s is the sum of the filtered array
# show((np.where(s > 255, s-255, 0) * 255/28).astype(np.uint8))