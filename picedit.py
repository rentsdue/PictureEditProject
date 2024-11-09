import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import time 

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

def distance(image, pix1, pix2):
    p1 = image[pix1[0], pix1[1]]
    p2 = image[pix2[0], pix2[1]]
    deltaRed = p1[0] - p2[0]
    deltaGreen = p1[1] - p2[1]
    deltaBlue = p1[2] - p2[2]
    redAvg = (p1[0] + p2[0]) / 2
    return math.sqrt((2 + redAvg / 256) * (deltaRed ** 2) + 4 * (deltaGreen ** 2) + (2 + (255 - redAvg) / 256) * (deltaBlue ** 2))

def magic_wand_select(image, x, thres):                
    row, col = np.shape(image)[:2]
    stack = [x]
    visitedList = []
    neighbour_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while len(stack) > 0:
        current_pix = stack.pop()
        if current_pix not in visitedList:
            visitedList.append(current_pix)
            for direction in neighbour_directions:
                nb = (current_pix[0] + direction[0], current_pix[1] + direction[1])
                if (0 <= nb[0] < row and 0 <= nb[1] < col and 
                    distance(image, nb, x) <= thres and 
                    nb not in visitedList):
                    stack.append(nb)
                    
    return create_mask(visitedList, row, col)


def create_mask(visitedList, row, col):
    msk = np.zeros((row, col), dtype=int)
    for pix in visitedList:
        msk[pix[0], pix[1]] = 1
    return msk

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

def applyMask(newImage, originalImage, mask):
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
            print("Thank you very much for using this picture editor. Have a nice day!")
            break
        elif userSelect == "l":
            while True:
                try:
                    filename = input("Enter the filename to load: ")
                    start_time = time.time()
                    image, mask = load_image(filename)
                    newImg = image.copy()
                    display_image(image, mask)
                    break
                except Exception as e:
                    print(f"An error occurred: {e}. Please enter a valid file name. You might have forgotten to input \".jpg\" or \".png\" in your file name.")
            
            end_time = time.time()
            print(f"Image loaded in {end_time - start_time:.4f} seconds.")
        
        elif userSelect == "s" and image is not None:
            start_time = time.time()
            newFileName = input("Enter the name of your new file: ")
            while True:
                imgFormat = input("What would you like to save your format as? Click 1 for .jpg, Click 2 for .png! ")
                if imgFormat == "1":
                    imgFormat = ".jpg"
                    break
                elif imgFormat == "2":
                    imgFormat = ".png"
                    break
                else:
                    print("Invalid input. Please try again!")
            save_image(newFileName + imgFormat, newImg.astype('uint8'))
            end_time = time.time()
            print(f"Image saved in {end_time - start_time:.4f} seconds.")
        
        elif userSelect == "1" and image is not None:
            while True:
                try:
                    rgbValue = int(
                        input("Enter an input value to change the image brightness (has to be an integer): "))
                    if rgbValue < -250 or rgbValue > 250:
                        print("Please input an integer between -250 and 250")
                        continue
                    break
                except ValueError:
                    print("Invalid input. Please enter an integer.")
            start_time = time.time()
            modifiedImg = change_brightness(newImg, rgbValue)
            if useNewMask:
                newImg = applyMask(modifiedImg, newImg, newMask) 
            else:
                newImg = modifiedImg
            end_time = time.time()
            print(f"Brightness adjusted in {end_time - start_time:.4f} seconds.")
            display_image(newImg, mask)
        
        elif userSelect == "2" and image is not None:
            while True:
                try:
                    contrastValue = int(input("Enter an input value to change the image contrast (has to be an integer): "))
                    break 
                except ValueError:
                    print("Invalid input. Please enter an integer.")
            start_time = time.time()
            modifiedImg = change_contrast(newImg, contrastValue)
            if useNewMask:
                newImg = applyMask(modifiedImg, newImg, newMask) 
            else:
                newImg = modifiedImg
            end_time = time.time()
            print(f"Contrast adjusted in {end_time - start_time:.4f} seconds.")
            display_image(newImg, mask)
        
        elif userSelect == "3" and image is not None:
            start_time = time.time()
            modifiedImg = grayscale(newImg)
            if useNewMask:
                newImg = applyMask(modifiedImg, newImg, newMask) 
            else:
                newImg = modifiedImg
            end_time = time.time()
            print(f"Grayscale applied in {end_time - start_time:.4f} seconds.")
            display_image(newImg, mask)
        
        elif userSelect == "4" and image is not None:
            start_time = time.time()
            modifiedImg = blur_effect(newImg)
            if useNewMask:
                newImg = applyMask(modifiedImg, newImg, newMask)  
            else:
                newImg = modifiedImg
            end_time = time.time()
            print(f"Blur applied in {end_time - start_time:.4f} seconds.")
            display_image(newImg, mask)
        
        elif userSelect == "5" and image is not None:
            start_time = time.time()
            modifiedImg = edge_detection(newImg)
            if useNewMask:
                newImg = applyMask(modifiedImg, newImg, newMask) 
            else:
                newImg = modifiedImg
            end_time = time.time()
            print(f"Edge detection applied in {end_time - start_time:.4f} seconds.")
            display_image(newImg, mask)
        
        elif userSelect == "6" and image is not None:
            start_time = time.time()
            modifiedImg = embossed(newImg)
            if (useNewMask):
                newImg = applyMask(modifiedImg, newImg, newMask)  
            else:
                newImg = modifiedImg
            end_time = time.time()
            print(f"Embossing applied in {end_time - start_time:.4f} seconds.")
            display_image(newImg, mask)
        
        elif userSelect == "7" and image is not None:
            while True:
                try:
                    while True:
                        x1 = int(input(f"Enter the x-coordinate of the top left corner of the rectangle (Must be an integer between 0 to {image.shape[1] - 1}): "))
                        if x1 < 0:
                            print("x-coordinate must be non-negative. Please try again.")
                        elif x1 >= image.shape[1]:
                            print(f"Coordinates must be within the dimensions of the image, which are {image.shape[1]} x {image.shape[0]}. Please try again.")
                        else:
                            break

                    while True:
                        y1 = int(input(f"Enter the y-coordinate of the top left corner of the rectangle (Must be an integer between 0 to {image.shape[0] - 1}): "))
                        if y1 < 0:
                            print("y-coordinate must be non-negative. Please try again.")
                        elif y1 >= image.shape[0]:
                            print(f"Coordinates must be within the dimensions of the image, which are {image.shape[1]} x {image.shape[0] - 1}. Please try again.")
                        else:
                            break

                    while True:
                        x2 = int(input(f"Enter the x-coordinate of the bottom right corner of the rectangle (Must be an integer between {x1} and {image.shape[1]}): "))
                        if x2 < 0:
                            print("x-coordinate must be non-negative. Please try again.")
                        elif x2 > image.shape[1]:
                            print(f"Coordinates must be within the dimensions of the image, which are {image.shape[1]} x {image.shape[0]}. Please try again.")
                        elif x2 <= x1:
                            print(f"Bottom right corner must be to the RIGHT of the top left corner (coordinate: {x1}, {y1}). Please try again.")
                        else:
                            break

                    while True:
                        y2 = int(input(f"Enter the y-coordinate of the bottom right corner of the rectangle (Must be an integer between {y1} and {image.shape[0]}): "))
                        if y2 < 0:
                            print("y-coordinate must be non-negative. Please try again.")
                        elif y2 > image.shape[0]:
                            print(f"Coordinates must be within the dimensions of the image, which are {image.shape[1]} x {image.shape[0]}. Please try again.")
                        elif y2 <= y1:
                            print(f"Bottom right corner must be to the BOTTOM of the top left corner (coordinate: {x1}, {y1}). Please try again.")
                        else:
                            break
                    break                 
                except ValueError:
                    print("Invalid input. Please enter an integer value.")

            start_time = time.time()
            top = (x1, y1)
            bottom = (x2, y2)
            newMask = rectangle_select(image, top, bottom)
            useNewMask = True
            end_time = time.time()
            print(f"Rectangle selected in {end_time - start_time:.4f} seconds.")
            display_image(newImg, newMask)

        
        elif userSelect == "8" and image is not None:
            while True:
                try:
                    while True:
                        xCoord = int(input(f"Please enter an x-coordinate (Must be between 0 and {image.shape[1] - 1}): "))
                        if xCoord < 0:
                            print("Coordinates must be non-negative. Please try again.")
                        elif xCoord >= image.shape[1]:
                            print(f"Coordinates must be within the dimensions of the image, which are {image.shape[1]} x {image.shape[0]}. Please try again.")
                        else:
                            break

                    while True:
                        yCoord = int(input(f"Please enter a y-coordinate (Must be between 0 and {image.shape[0] - 1}): "))
                        if yCoord < 0:
                            print("Coordinates must be non-negative. Please try again.")
                        elif yCoord >= image.shape[0]:
                            print(f"Coordinates must be within the dimensions of the image, which are {image.shape[1]} x {image.shape[0]}. Please try again.")
                        else:
                            break
                        
                    thres = int(input("Please enter a threshold: "))
                    if thres < 0:
                        print("The threshold value must be non-negative. Please try again.")
                        continue 
                    break  
                except ValueError:
                    print("Invalid input. Please enter an integer value.")
            start_time = time.time()
            newMask = magic_wand_select(image, (xCoord, yCoord), thres)
            useNewMask = True
            end_time = time.time()
            print(f"Magic wand selection applied in {end_time - start_time:.4f} seconds.")
            display_image(newImg, newMask)
        
        else:
            print("Invalid choice. Please try again.")
            continue

if __name__ == "__main__":
    menu()






