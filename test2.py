import numpy as np
from picedit import *

def test():
    # ***************** copy test ***************** #
    img, _ = load_image("mini_test.png")
    img_cpr = img.copy()

    # Apply multiple functions and verify no modification to original image
    _ = change_brightness(img, 10)
    _ = change_contrast(img, 10)
    _ = grayscale(img)
    _ = blur_effect(img)
    _ = edge_detection(img)
    _ = embossed(img)
    _ = rectangle_select(img, (0, 0), (1, 1))
    _ = magic_wand_select(img, (0, 0), 0)

    if np.array_equal(img_cpr, img):
        print("test copy - OK!")
    else:
        print("test copy - problem: input image is modified in one of your functions!")
        print("Skipping other tests")
        return -1

    # ***************** brightness test ***************** #
    img, _ = load_image("mini_test.png")
    image = change_brightness(img, 10)
    
    # Edge case: Brightness adjustment on a completely dark image (should increase values)
    img_dark = np.zeros((5, 5, 3), dtype=np.uint8)
    image = change_brightness(img_dark, 50)
    if np.all(image == 50):
        print("test brightness - dark image brightened - OK!")
    else:
        print("test brightness - Problem with dark image brightening!")
    
    # Edge case: Brightness adjustment on a completely bright image (should clamp values)
    img_bright = np.full((5, 5, 3), 255, dtype=np.uint8)
    image = change_brightness(img_bright, 50)
    if np.all(image == 255):
        print("test brightness - bright image clamped - OK!")
    else:
        print("test brightness - Problem with bright image clamping!")

    # ***************** contrast test ***************** #
    img, _ = load_image("mini_test.png")
    image = change_contrast(img, 50)

    # Edge case: Contrast adjustment on a mid-level gray image
    img_gray = np.full((5, 5, 3), 128, dtype=np.uint8)
    image = change_contrast(img_gray, 50)
    if np.all(image == 128):
        print("test contrast - mid-gray image unchanged - OK!")
    else:
        print("test contrast - Problem with mid-gray image contrast adjustment!")

    # ***************** grayscale test ***************** #
    img, _ = load_image("mini_test.png")
    image = grayscale(img)
    
    # Edge case: Grayscale a single-color image (output should remain uniform)
    img_single_color = np.full((5, 5, 3), [123, 200, 50], dtype=np.uint8)
    image = grayscale(img_single_color)
    if np.all(image[:, :, 0] == image[:, :, 1]) and np.all(image[:, :, 1] == image[:, :, 2]):
        print("test grayscale - single-color image - OK!")
    else:
        print("test grayscale - Problem with single-color image grayscaling!")

    # ***************** blur_effect test ***************** #
    img, _ = load_image("mini_test.png")
    image = blur_effect(img)

    # Edge case: Blur on an image with sharp edges (should smooth out values)
    img_sharp_edges = np.zeros((5, 5, 3), dtype=np.uint8)
    img_sharp_edges[2, 2] = [255, 255, 255]  # Single white pixel in the middle
    image = blur_effect(img_sharp_edges)
    if np.all(image[2, 2] < [255, 255, 255]) and np.any(image != 0):
        print("test blur_effect - sharp edge smoothing - OK!")
    else:
        print("test blur_effect - Problem with sharp edge smoothing!")

    # ***************** edge_detection test ***************** #
    img, _ = load_image("mini_test.png")
    image = edge_detection(img)

    # Edge case: Edge detection on a uniform image (no edges should be detected)
    img_uniform = np.full((5, 5, 3), 150, dtype=np.uint8)
    image = edge_detection(img_uniform)
    if np.all(image == 0):
        print("test edge_detection - uniform image - OK!")
    else:
        print("test edge_detection - Problem with uniform image edge detection!")

    # ***************** embossed test ***************** #
    img, _ = load_image("mini_test.png")
    image = embossed(img)

    # Edge case: Emboss on a gradient image (should highlight differences)
    img_gradient = np.array([[[i * 51, j * 51, 0] for j in range(5)] for i in range(5)], dtype=np.uint8)
    image = embossed(img_gradient)
    if np.any(image != img_gradient):
        print("test embossed - gradient image - OK!")
    else:
        print("test embossed - Problem with gradient image embossing!")

    # ***************** rectangle_select test ***************** #
    img, _ = load_image("mini_test.png")
    mask = rectangle_select(img, (1, 1), (3, 4))

    # Edge case: Rectangle selection for a single pixel
    mask = rectangle_select(img, (2, 2), (2, 2))
    if np.count_nonzero(mask) == 1:
        print("test rectangle_select - single pixel - OK!")
    else:
        print("test rectangle_select - Problem with single pixel selection!")

    # ***************** magic_wand_select test ***************** #
    img, _ = load_image("mini_test.png")
    mask = magic_wand_select(img, (1, 1), 300)

    # Edge case: Magic wand with maximum threshold (should select entire image)
    mask = magic_wand_select(img, (2, 2), 1000)
    if np.sum(mask) == img.shape[0] * img.shape[1]:
        print("test magic_wand_select - max threshold - OK!")
    else:
        print("test magic_wand_select - Problem with max threshold selection!")

    # Edge case: Magic wand with minimum threshold (should select only the start pixel)
    mask = magic_wand_select(img, (2, 2), 0)
    if np.sum(mask) == 1:
        print("test magic_wand_select - min threshold - OK!")
    else:
        print("test magic_wand_select - Problem with min threshold selection!")

test()
