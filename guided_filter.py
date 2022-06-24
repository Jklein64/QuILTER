import numpy as np
from PIL import Image

# determined through experimentation
GUIDED_FILTER_RADIUS = 30
# taken from example on GitHub
GUIDED_FILTER_EPSILON = 1e-6 * 255 ** 2

# note: adding with a limit and then adding the rest
# to the next item in the array until all of the values
# have been added is a really hard problem to solve with
# raw numpy arrays.  Why?

def main():
    # loads everything as 8-bit images
    image = np.array(Image.open("original.png"))
    masks = [np.array(Image.open(f"{i}.png")) for i in (1, 2, 3)]
    filtered = filter(np.stack(masks, axis=-1), image).astype(int)

    # scale to keep the ratio between masks but have a max sum of 255
    scaled = (255 * filtered / np.sum(filtered, axis=-1, keepdims=True)).astype(np.uint8)
    ascending_i = np.argsort(scaled, axis=-1)
    union = np.sum(scaled, axis=-1)
    # absolute value to keep as uint8; direction is from union > 255
    difference = np.absolute(union.astype(int) - 255).astype(np.uint8)
    # create array to add the difference to the largest entry
    addend = np.where(ascending_i[union < 255] == len(masks) - 1, difference[union < 255, np.newaxis], 0)
    # TODO doesn't currently account for integer overflow. is that even possible?
    scaled[union < 255] += addend

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