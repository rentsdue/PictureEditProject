import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time # Remove at the very end, use it to test how long it takes for the function to run

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
    x_i = x[0]
    x_f = x[1]
    y_i = y[0]
    y_f = y[1]

    rect = np.zeros((np.shape(image)[0], np.shape(image)[1]))   
    rect[x_i:y_i+1, x_f:y_f+1] = 1
    print(rect)
    return rect

def magic_wand_select(image, x, thres):                
    row, col = np.shape(image)[:2]
    stack = []
    stack.append(x)
    visited_lst = []

    while len(stack) > 0:
        current_pix = stack.pop()
        visited_lst.append(current_pix)
        valid_neighbours = finding_valid_neighbours(image, current_pix, visited_lst, thres,x)
        stack.extend(valid_neighbours)
    return create_mask(visited_lst, row, col)

def finding_valid_neighbours(image, current_pix, visited_lst, thres,x):
    row, col,_ = image.shape
    neighbour_direction = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    valid_neighbours = []

    for direction in neighbour_direction:
        nb = (current_pix[0] + direction[0], current_pix[1] + direction[1])
        if is_indeed_valid_neighbour(nb, row, col, image, current_pix, thres, visited_lst,x):
            valid_neighbours.append(nb)
    return valid_neighbours

def is_indeed_valid_neighbour(nb, row, col, image, current_pix, thres, visited_lst,x):
    return (0 <= nb[0] < row and 0 <= nb[1] < col and 
distance(image, nb, x) <= thres and 
            nb not in visited_lst)

def create_mask(visited_lst, row, col):
    msk = np.zeros((row, col), dtype=int)
    for pix in visited_lst:
        msk[pix[0], pix[1]] = 1
    return msk

import math
def distance(image, pix1, pix2):
    p1=image[pix1[0],pix1[1]]
    p2=image[pix2[0],pix2[1]]
    dr=p1[0]-p2[0]
    dg=p1[1]-p2[1]
    db=p1[2]-p2[2]
    r=(p1[0]+p2[0])/2
    return math.sqrt((2+r/256)*(dr**2)+4*(dg**2)+(2+(255-r)/256)*(db**2))


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

def apply_mask(newImage, originalImage, mask):
    finalImage = originalImage.copy()
    for i in range(len(originalImage)):
        for j in range(len(originalImage[i])):
            if(mask[i][j]==1):
                finalImage[i][j]= newImage[i][j]
            else:
                pass
        return finalImage

def menu():
    image = None
    newImg = None
    mask = None
    newMask = None
    useNewMask = False

    while True:
        if image is None:
            userSelect = input("What do you want to do ? \n e - exit \n l - load a picture \n \n Your choice: ")
        else: 
            userSelect = input("What do you want to do ? \n e - exit \n l - load a picture \n s - save the current picture \n 1 - adjust brightness \n 2 - adjust contrast \n 3 - apply grayscale \n 4 - apply blur \n 5 - edge detection \n 6 - embossed \n 7 - rectangle select \n 8 - magic wand select \n \n Your choice: ")
        
        if userSelect == "e":
            break
        elif userSelect == "l":
            filename = input("Enter the filename to load: ")
            start_time = time.time()
            image, mask = load_image(filename)
            newImg = image.copy()  # Ensure `newImg` is a copy of the loaded image
            end_time = time.time()
            print(f"Image loaded in {end_time - start_time:.4f} seconds.")
        
        elif userSelect == "s" and image is not None:
            start_time = time.time()
            save_image(filename, newImg)  # Save the modified image
            end_time = time.time()
            print(f"Image saved in {end_time - start_time:.4f} seconds.")
        
        elif userSelect == "1" and image is not None:
            rgbValue = int(input("Enter an input value to change the image brightness: "))
            start_time = time.time()
            modifiedImg = change_brightness(newImg, rgbValue)
            newImg = apply_mask(newImg, modifiedImg, newMask) if useNewMask and newMask is not None else modifiedImg
            end_time = time.time()
            print(f"Brightness adjusted in {end_time - start_time:.4f} seconds.")
            display_image(newImg, newMask if useNewMask else mask)
        
        elif userSelect == "2" and image is not None:
            contrastValue = int(input("Enter an input value to change the image contrast: "))
            start_time = time.time()
            modifiedImg = change_contrast(newImg, contrastValue)
            newImg = apply_mask(newImg, modifiedImg, newMask) if useNewMask and newMask is not None else modifiedImg
            end_time = time.time()
            print(f"Contrast adjusted in {end_time - start_time:.4f} seconds.")
            display_image(newImg, newMask if useNewMask else mask)
        
        elif userSelect == "3" and image is not None:
            start_time = time.time()
            modifiedImg = grayscale(newImg)
            newImg = apply_mask(newImg, modifiedImg, newMask) if useNewMask and newMask is not None else modifiedImg
            end_time = time.time()
            print(f"Grayscale applied in {end_time - start_time:.4f} seconds.")
            display_image(newImg, newMask if useNewMask else mask)
        
        elif userSelect == "4" and image is not None:
            start_time = time.time()
            modifiedImg = blur_effect(newImg)
            newImg = apply_mask(newImg, modifiedImg, newMask) if useNewMask and newMask is not None else modifiedImg
            end_time = time.time()
            print(f"Blur applied in {end_time - start_time:.4f} seconds.")
            display_image(newImg, newMask if useNewMask else mask)
        
        elif userSelect == "5" and image is not None:
            start_time = time.time()
            modifiedImg = edge_detection(newImg)
            newImg = apply_mask(newImg, modifiedImg, newMask) if useNewMask and newMask is not None else modifiedImg
            end_time = time.time()
            print(f"Edge detection applied in {end_time - start_time:.4f} seconds.")
            display_image(newImg, newMask if useNewMask else mask)
        
        elif userSelect == "6" and image is not None:
            start_time = time.time()
            modifiedImg = embossed(newImg)
            newImg = apply_mask(newImg, modifiedImg, newMask) if useNewMask and newMask is not None else modifiedImg
            end_time = time.time()
            print(f"Embossing applied in {end_time - start_time:.4f} seconds.")
            display_image(newImg, newMask if useNewMask else mask)
        
        elif userSelect == "7" and image is not None:
            x1 = int(input("Enter the x-coordinate of the top left corner of the rectangle: "))
            y1 = int(input("Enter the y-coordinate of the top left corner of the rectangle: "))
            x2 = int(input("Enter the x-coordinate of the bottom right corner of the rectangle: "))
            y2 = int(input("Enter the y-coordinate of the bottom right corner of the rectangle: "))

            if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
                raise ValueError("Coordinates must be non-negative.")
            if x1 >= x2:
                raise ValueError("Top left corner must be to the LEFT of the bottom right corner.")
            if y2 <= y1:
                raise ValueError("Top left corner must be ABOVE the bottom right corner.")
            if x1 > image.shape[1] or y1 > image.shape[0] or x2 > image.shape[1] or y2 > image.shape[0]:
                raise ValueError("Coordinates must be within the dimensions of the image.")

            start_time = time.time()
            top = (x1, y1)
            bottom = (x2, y2)
            newMask = rectangle_select(image, top, bottom)
            useNewMask = True
            end_time = time.time()
            print(f"Rectangle selected in {end_time - start_time:.4f} seconds.")
            display_image(newImg, newMask)
        
        elif userSelect == "8" and image is not None:
            xCoord = int(input("Please enter an x-coordinate: "))
            yCoord = int(input("Please enter a y-coordinate: "))
            thres = int(input("Please enter a threshold: "))
            start_time = time.time()
            newMask = magic_wand_select(image, (xCoord, yCoord), thres)
            useNewMask = True
            end_time = time.time()
            print(f"Magic wand selection applied in {end_time - start_time:.4f} seconds.")
            display_image(newImg, newMask)
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()






