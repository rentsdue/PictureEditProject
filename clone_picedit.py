import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def change_brightness(image, value):
    brightImg = image.copy()
    for i in range(len(image)): 
        for j in range(len(image[i])):
            for k in range(len(image[i][j])):
                brightImg[i][j][k] += value
                if (brightImg[i][j][k] > 255):
                    brightImg[i][j][k] = 255
                if (brightImg[i][j][k] < 0):
                    brightImg[i][j][k] = 0

    return brightImg

def change_contrast(image, value):
    contrastImg = image.copy()
    factor = (259 * (value + 255)) / (255 * (259 - value))
    for i in range(len(image)): 
        for j in range(len(image[i])):
            for k in range(len(image[i][j])):
                contrastImg[i][j][k] = factor * (image[i][j][k] - 128) + 128 
                if (contrastImg[i][j][k] > 255): 
                    contrastImg[i][j][k] = 255
                elif (contrastImg[i][j][k] < 0):
                    contrastImg[i][j][k] = 0
    return contrastImg

def grayscale(image):
    grayImg = image.copy()
    for i in range(len(image)):
        for j in range(len(image[i])):
            red = grayImg[i][j][0]
            green = grayImg[i][j][1]
            blue = grayImg[i][j][2]
            grayscaled = int(0.3 * red + 0.59 * green + 0.11 * blue)
            for k in range(len(image[i][j])):
                grayImg[i][j][k] = grayscaled
    return grayImg 

def blur_effect(image):
    blurredImg = image.copy()
    for i in range(1, len(image) - 1): 
        for j in range(1, len(image[i]) - 1):
            for k in range(len(image[i][j])):
                blurredImg[i][j][k] = 0.0625 * image[i - 1][j - 1][k] + 0.125 * image[i - 1][j][k] + 0.0625 * image[i - 1][j + 1][k]  + 0.125 * image[i][j - 1][k]  + 0.25 * image[i][j][k] + 0.125 * image[i][j + 1][k]+ 0.0625 * image[i + 1][j - 1][k] + 0.125 * image[i + 1][j][k] + 0.0625 * image[i + 1][j + 1][k]
                if (blurredImg[i][j][k] > 255):  
                    blurredImg[i][j][k] = 255
                elif (blurredImg[i][j][k] < 0):
                    blurredImg[i][j][k] = 0
    return blurredImg 

def edge_detection(image):
    newImg = image.copy()
    for i in range(1, len(image) - 1): 
        for j in range(1, len(image[i]) - 1):
            for k in range(len(image[i][j])):
                newImg[i][j][k] =  (-1 * image[i - 1][j - 1][k]) + (-1 * image[i - 1][j][k]) + (-1 * image[i - 1][j + 1][k] ) + (-1 * image[i][j - 1][k]) + (8 * image[i][j][k]) + (-1 * image[i][j + 1][k]) + (-1 * image[i + 1][j - 1][k]) + (-1 * image[i + 1][j][k]) + (-1 * image[i + 1][j + 1][k]) 
                newImg[i][j][k] += 128
                if (newImg[i][j][k] > 255):  
                    newImg[i][j][k] = 255
                elif (newImg[i][j][k] < 0):
                    newImg[i][j][k] = 0
    return newImg 

def embossed(image):
    newImg = image.copy()
    for i in range(1, len(image) - 1): 
        for j in range(1, len(image[i]) - 1):
            for k in range(len(image[i][j])):
                newImg[i][j][k] =  (-1 * image[i - 1][j - 1][k]) + (-1 * image[i - 1][j][k]) + (-1 * image[i][j - 1][k]) + (image[i][j + 1][k]) + (image[i + 1][j][k]) + (image[i + 1][j + 1][k]) 
                newImg[i][j][k] += 128
                if (newImg[i][j][k] > 255):  
                    newImg[i][j][k] = 255
                elif (newImg[i][j][k] < 0):
                    newImg[i][j][k] = 0
    return newImg 

def rectangle_select(image, x, y):
    return np.array([]) # to be removed when filling this function

def magic_wand_select(image, x, thres):                
    return np.array([]) # to be removed when filling this function

def compute_edge(mask):           
    rsize, csize = len(mask), len(mask[0]) 
    edge = np.zeros((rsize,csize))
    if np.all((mask == 1)): return edge        
    for r in range(rsize):
        for c in range(csize):
            if mask[r][c]!=0:
                if r==0 or c==0 or r==len(mask)-1 or c==len(mask[0])-1:
                    edge[r][c]=1
                    continue
                
                is_edge = False                
                for var in [(-1,0),(0,-1),(0,1),(1,0)]:
                    r_temp = r+var[0]
                    c_temp = c+var[1]
                    if 0<=r_temp<rsize and 0<=c_temp<csize:
                        if mask[r_temp][c_temp] == 0:
                            is_edge = True
                            break
    
                if is_edge == True:
                    edge[r][c]=1
            
    return edge

def save_image(filename, image):
    img = image.astype(np.uint8)
    mpimg.imsave(filename,img)

def load_image(filename):
    img = mpimg.imread(filename)
    if len(img[0][0])==4: # if png file
        img = np.delete(img, 3, 2)
    if type(img[0][0][0])==np.float32:  # if stored as float in [0,..,1] instead of integers in [0,..,255]
        img = img*255
        img = img.astype(np.uint8)
    mask = np.ones((len(img),len(img[0]))) # create a mask full of "1" of the same size of the laoded image
    img = img.astype(np.int32)
    return img, mask

def display_image(image, mask):
    # if using Spyder, please go to "Tools -> Preferences -> IPython console -> Graphics -> Graphics Backend" and select "inline"
    tmp_img = image.copy()
    edge = compute_edge(mask)
    for r in range(len(image)):
        for c in range(len(image[0])):
            if edge[r][c] == 1:
                tmp_img[r][c][0]=255
                tmp_img[r][c][1]=0
                tmp_img[r][c][2]=0
 
    plt.imshow(tmp_img)
    plt.axis('off')
    plt.show()
    print("Image size is",str(len(image)),"x",str(len(image[0])))


def menu():
    image = None
    mask = None

    while True:
        if image is None:
            userSelect = input("What do you want to do ? \n e - exit \n l - load a picture \n \n Your choice: ")
        else: 
            userSelect = input("What do you want to do ? \n e - exit \n l - load a picture \n s - save the current picture \n 1 - adjust brightness \n 2 - adjust contrast \n 3 - apply grayscale \n 4 - apply blur \n 5 - edge detection \n 6 - embossed \n 7 - rectangle select \n 8 - magic wand select \n \n Your choice: ")

        if userSelect == "e":
            break
        elif userSelect == "l":
            filename = input("Enter the filename to load: ")
            image, mask = load_image(filename)
        elif userSelect == "s" and image is not None:
            save_image()
        elif userSelect == "1" and image is not None:
            rgbValue = int(input("Enter an input value in order to change the image brightness: "))
            newImg = change_brightness(image, rgbValue)
            display_image(newImg, mask)
        elif userSelect == "2" and image is not None:
            contrastValue = int(input("Enter an input value in order to change the image contrast: "))
            newImg = change_contrast(image, contrastValue)
            display_image(newImg, mask)
        elif userSelect == "3" and image is not None:
            newImg = grayscale(image)
            display_image(newImg, mask)
        elif userSelect == "4" and image is not None:
            newImg = blur_effect(image)
            display_image(newImg, mask)
        elif userSelect == "5" and image is not None:
            newImg = edge_detection(image)
            display_image(newImg, mask)
        elif userSelect == "6" and image is not None:
            newImg = embossed(image)
            display_image(newImg, mask)
        elif userSelect == "7" and image is not None:
            rectangle_select()
        elif userSelect == "8" and image is not None:
            magic_wand_select()
        else:
            print("Invalid choice. Please try again.")  

if __name__ == "__main__":
    menu()
