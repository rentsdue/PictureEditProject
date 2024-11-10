import numpy as np
from picedit import *

def test():
    # ***************** copy test ***************** #
    img, _ = load_image("mini_test.png")
    img_cpr = np.array([91 ,0 ,0 ,155 ,32 ,113 ,74 ,0 ,177 ,174 ,139 ,255 ,33 ,66 ,145 ,84 ,69 ,234 ,189 ,157 ,254 ,78 ,24 ,60 ,170 ,101 ,145 ,81 ,3 ,105 ,110 ,99 ,0 ,182 ,218 ,172 ,0 ,28 ,97 ,229 ,173 ,255 ,217 ,24 ,87 ,108 ,121 ,16 ,223 ,215 ,108 ,237 ,243 ,137 ,189 ,255 ,175 ,56 ,245 ,145 ,82 ,153 ,145 ,49 ,100 ,0 ,140 ,187 ,23 ,99 ,177 ,91 ,18 ,147 ,202]).reshape(5,5,3)
    image = change_brightness(img,10)
    image = change_contrast(img,10)
    image = grayscale(img)
    image = blur_effect(img)
    image = edge_detection(img)
    image = embossed(img)
    mask = rectangle_select(img, (0,0), (1,1))
    mask = magic_wand_select(img, (0,0), 0)
    if np.array_equal(img_cpr,img): print("test copy - OK !")
    else: 
        print("test copy - problem: input image is modified in one of your functions !")
        print("Skipping the other tests")
        return -1

    # Edge case: no modifications should happen to original image in functions
    # Additional copy test
    img_copy_test = img.copy()
    _ = change_brightness(img_copy_test, 10)
    if not np.array_equal(img, img_copy_test):
        print("test copy - problem: function modified the input image!")
        return -1
    
    # ***************** brightness test ***************** #
    # Normal brightness adjustment
    img, _ = load_image("mini_test.png")
    img_cpr = np.array([101 ,10 ,10 ,165 ,42 ,123 ,84 ,10 ,187 ,184 ,149 ,255 ,43 ,76 ,155 ,94 ,79 ,244 ,199 ,167 ,255 ,88 ,34 ,70 ,180 ,111 ,155 ,91 ,13 ,115 ,120 ,109 ,10 ,192 ,228 ,182 ,10 ,38 ,107 ,239 ,183 ,255 ,227 ,34 ,97 ,118 ,131 ,26 ,233 ,225 ,118 ,247 ,253 ,147 ,199 ,255 ,185 ,66 ,255 ,155 ,92 ,163 ,155 ,59 ,110 ,10 ,150 ,197 ,33 ,109 ,187 ,101 ,28 ,157 ,212]).reshape(5,5,3)
    image = change_brightness(img,10)
    if len(image)==0 or not np.allclose(img_cpr,image,atol=1): print("test brightness - Problem in the brightness function output !")
    else: print("test brightness - OK !")

    # Edge case: maximum brightness adjustment
    image = change_brightness(img, 255)
    if np.any(image > 255) or np.any(image < 0): print("test brightness - Problem: values exceed RGB bounds!")
    
    # Edge case: minimum brightness adjustment
    image = change_brightness(img, -255)
    if np.any(image > 255) or np.any(image < 0): print("test brightness - Problem: values exceed RGB bounds!")

    # Invalid input type
    try:
        image = change_brightness(img, "high")
        print("test brightness - Problem: did not handle invalid input!")
    except:
        print("test brightness - Invalid input handled correctly!")

    # ***************** contrast test ***************** #
    img, _ = load_image("mini_test.png")
    img_cpr = np.array([73 ,0 ,0 ,168 ,0 ,105 ,47 ,0 ,200 ,196 ,144 ,255 ,0 ,36 ,153 ,62 ,40 ,255 ,218 ,170 ,255 ,53 ,0 ,27 ,190 ,87 ,153 ,58 ,0 ,93 ,101 ,85 ,0 ,208 ,255 ,193 ,0 ,0 ,82 ,255 ,194 ,255 ,255 ,0 ,67 ,98 ,117 ,0 ,255 ,255 ,98 ,255 ,255 ,141 ,218 ,255 ,197 ,21 ,255 ,153 ,59 ,165 ,153 ,10 ,86 ,0 ,145 ,215 ,0 ,85 ,200 ,73 ,0 ,156 ,237 , ]).reshape(5,5,3)
    image = change_contrast(img,50)
    if len(image)==0 or not np.allclose(img_cpr,image,atol=1): print("test contrast - Problem in the contrast function output !")
    else: print("test contrast - OK !")

    # Edge case: maximum contrast
    image = change_contrast(img, 255)
    if np.any(image > 255) or np.any(image < 0): print("test contrast - Problem: values exceed RGB bounds!")

    # Edge case: minimum contrast
    image = change_contrast(img, -255)
    if np.any(image > 255) or np.any(image < 0): print("test contrast - Problem: values exceed RGB bounds!")

    # Invalid input
    try:
        image = change_contrast(img, "low")
        print("test contrast - Problem: did not handle invalid input!")
    except:
        print("test contrast - Invalid input handled correctly!")

    # ***************** grayscale test ***************** #
    img, _ = load_image("mini_test.png")
    img_cpr = np.array([27 ,27 ,27 ,77 ,77 ,77 ,41 ,41 ,41 ,162 ,162 ,162 ,64 ,64 ,64 ,91 ,91 ,91 ,177 ,177 ,177 ,44 ,44 ,44 ,126 ,126 ,126 ,37 ,37 ,37 ,91 ,91 ,91 ,202 ,202 ,202 ,27 ,27 ,27 ,198 ,198 ,198 ,88 ,88 ,88 ,105 ,105 ,105 ,205 ,205 ,205 ,229 ,229 ,229 ,226 ,226 ,226 ,177 ,177 ,177 ,130 ,130 ,130 ,73 ,73 ,73 ,154 ,154 ,154 ,144 ,144 ,144 ,114 ,114 ,114 , ]).reshape(5,5,3)
    image = grayscale(img)
    if len(image)==0 or not (img_cpr.size==image.size and np.allclose(img_cpr,image,atol=1)): print("test grayscale - Problem in the grayscale function output !")
    else: print("test grayscale - OK !")

    # Edge case: grayscale on already grayscale image
    image = grayscale(img_cpr)
    if not np.allclose(image, img_cpr): print("test grayscale - Problem: grayscale did not maintain the same output!")

     # ***************** blur_effect test ***************** #
    img, _ = load_image("mini_test.png")
    img_cpr = np.array([91 ,0 ,0 ,155 ,32 ,113 ,74 ,0 ,177 ,174 ,139 ,255 ,33 ,66 ,145 ,84 ,69 ,234 ,126 ,90 ,153 ,119 ,76 ,148 ,133 ,75 ,152 ,81 ,3 ,105 ,110 ,99 ,0 ,142 ,145 ,128 ,138 ,134 ,144 ,157 ,126 ,154 ,217 ,24 ,87 ,108 ,121 ,16 ,148 ,168 ,84 ,163 ,188 ,117 ,148 ,192 ,147 ,56 ,245 ,145 ,82 ,153 ,145 ,49 ,100 ,0 ,140 ,187 ,23 ,99 ,177 ,91 ,18 ,147 ,202 , ]).reshape(5,5,3)    
    image = blur_effect(img)
    if len(image)==0 or not (img_cpr.size==image.size and np.allclose(img_cpr,image,atol=1)): print("test blur_effect - Problem in the blur_effect function output !") 
    else: print("test blur_effect - OK !") 

    # Edge case: blur effect on a very small image (1x1 pixel)
    img_small = np.array([[[123, 123, 123]]])
    image = blur_effect(img_small)
    if not np.array_equal(image, img_small): print("test blur_effect - Problem: unexpected result on 1x1 image!")
    
    # Edge case: multiple blurs for stronger effect
    img, _ = load_image("mini_test.png")
    for _ in range(5):  # Applying blur multiple times
        image = blur_effect(img)
    print("test blur_effect - Multiple blur effect applied (check visually for smoothness if possible).")

    # ***************** edge_detection test ***************** #
    img, _ = load_image("mini_test.png")
    img_cpr = np.array([91 ,0 ,0 ,155 ,32 ,113 ,74 ,0 ,177 ,174 ,139 ,255 ,33 ,66 ,145 ,84 ,69 ,234 ,255 ,255 ,255 ,0 ,0 ,0 ,255 ,255 ,107 ,81 ,3 ,105 ,110 ,99 ,0 ,255 ,255 ,255 ,0 ,0 ,0 ,255 ,255 ,255 ,217 ,24 ,87 ,108 ,121 ,16 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,56 ,245 ,145 ,82 ,153 ,145 ,49 ,100 ,0 ,140 ,187 ,23 ,99 ,177 ,91 ,18 ,147 ,202 , ]).reshape(5,5,3)
    image = edge_detection(img)
    if len(image)==0 or not (img_cpr.size==image.size and np.allclose(img_cpr,image,atol=1)): print("test edge_detection - Problem in the edge_detection function output !")
    else: print("test edge_detection - OK !") 

    # Edge case: edge detection on a plain, single-color image
    img_single_color = np.full((5, 5, 3), 150)
    image = edge_detection(img_single_color)
    if not np.array_equal(image, img_single_color): print("test edge_detection - Plain color edge case handled!")

    # ***************** embossed test ***************** #
    img, _ = load_image("mini_test.png")
    img_cpr = np.array([91 ,0 ,0 ,155 ,32 ,113 ,74 ,0 ,177 ,174 ,139 ,255 ,33 ,66 ,145 ,84 ,69 ,234 ,58 ,255 ,110 ,109 ,241 ,81 ,255 ,165 ,83 ,81 ,3 ,105 ,110 ,99 ,0 ,205 ,255 ,0 ,255 ,255 ,209 ,255 ,255 ,233 ,217 ,24 ,87 ,108 ,121 ,16 ,154 ,220 ,100 ,151 ,255 ,40 ,0 ,253 ,77 ,56 ,245 ,145 ,82 ,153 ,145 ,49 ,100 ,0 ,140 ,187 ,23 ,99 ,177 ,91 ,18 ,147 ,202 , ]).reshape(5,5,3)
    image = embossed(img)
    if len(image)==0 or not (img_cpr.size==image.size and np.allclose(img_cpr,image,atol=1)): print("test embossed - Problem in the embossed function output !")
    else: print("test embossed - OK !") 

    # Edge case: embossed effect on plain image
    img_single_color = np.full((5, 5, 3), 150)
    image = embossed(img_single_color)
    if not np.array_equal(image, img_single_color): print("test embossed - Plain color emboss effect handled!")

    # ***************** rectangle_select test ***************** #
    img, _ = load_image("mini_test.png")
    mask_cpr = np.array([0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,
                         0.0 ,1.0 ,1.0 ,1.0 ,1.0 ,
                         0.0 ,1.0 ,1.0 ,1.0 ,1.0 ,
                         0.0 ,1.0 ,1.0 ,1.0 ,1.0 ,
                         0.0 ,0.0 ,0.0 ,0.0 ,0.0 , ]).reshape(5,5)
    mask = rectangle_select(img,(1,1),(3,4))
    if len(mask)==0 or not np.array_equal(mask_cpr,mask): print("test rectangle_select - Problem in the rectangle_select function output !")
    else: print("test rectangle_select - OK !") 

    # Edge case: rectangle with invalid coordinates
    try:
        mask = rectangle_select(img, (10, 10), (3, 4))
        print("test rectangle_select - Problem: did not handle out-of-bounds input!")
    except:
        print("test rectangle_select - Invalid input handled correctly!")

    # Edge case: zero-area rectangle (same start and end points)
    mask = rectangle_select(img, (2, 2), (2, 2))
    if np.count_nonzero(mask) != 1: print("test rectangle_select - Problem: single point selection failed.")

    # ***************** magic_wand_select test ***************** #
    img, _ = load_image("mini_test.png")
    mask_cpr = np.array([0.0 ,0.0 ,0.0 ,1.0 ,0.0 ,
                         1.0 ,1.0 ,0.0 ,1.0 ,0.0 ,
                         0.0 ,1.0 ,0.0 ,1.0 ,0.0 ,
                         0.0 ,1.0 ,1.0 ,1.0 ,0.0 ,
                         0.0 ,0.0 ,0.0 ,1.0 ,1.0 , ]).reshape(5,5)
    mask = magic_wand_select(img,(1,1),300)
    if len(mask)==0 or not np.array_equal(mask_cpr,mask): print("test magic_wand_select - Problem in the magic_wand_select function output !")
    else: print("test magic_wand_select - OK !") 

    # Edge case: magic wand select with high threshold (should select many pixels)
    mask = magic_wand_select(img, (2, 2), 1000)
    if np.sum(mask) < img.size // 3: print("test magic_wand_select - Problem: high threshold selection too small.")

    # Edge case: magic wand select with low threshold (should select very few pixels)
    mask = magic_wand_select(img, (2, 2), 0)
    if np.sum(mask) > 1: print("test magic_wand_select - Problem: low threshold selection too large.")

    # Invalid input tests
    try:
        mask = magic_wand_select(img, (10, 10), "high")
        print("test magic_wand_select - Problem: did not handle invalid input!")
    except:
        print("test magic_wand_select - Invalid input handled correctly!")

test()
