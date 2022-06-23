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
    filtered = filter(np.stack(masks, axis=-1), image)

    union = np.sum(filtered, axis=-1)
    # prefer sparsity
    # subtract from smallest channel
    # add to largest channel

    # absolute value to keep as uint8; direction is from union > 255
    difference = np.absolute(union.astype(int) - 255).astype(np.uint8)

    # handle where the sum of the masks is too large
    smallest = np.argsort(filtered[union > 255], axis=1)
    # subtract from smallest channels first to increase sparsity
    # cycle to next channel at 0 to avoid integer underflow
    remaining = difference[union > 255]
    for i, channels in enumerate(filtered[union > 255]):
        channel = 0
        while remaining[i] > 0:
            value = channels[smallest[i][channel]]
            if value >= remaining[i]:
                filtered[union > 255][smallest[i][channel]] -= remaining[i]
                remaining[i] = 0
            else:
                remaining[i] -= value
                filtered[union > 255][smallest[i][channel]] = 0
                channel += 1


    # indices = np.zeros_like(remaining)
    # while remaining > 0:
    #     value = filtered[union > 255, smallest]
    #     if value >= remaining:
    #         filtered[union > 255, smallest] -= remaining
    #         remaining = 0
    #     else:


    # filtered[union > 255, smallest] -= difference[union > 255]

    # handle where the sum of the masks is too small
    largest = np.argmax(filtered[union < 255], axis=-1)
    # add to largest channel to increase sparsity
    filtered[union < 255, largest] += difference[union < 255]

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