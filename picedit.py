import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import time 

def change_brightness(image, value):
    rows, columns, colors = image.shape # Use this to get variables for iterating across the for loop (image.shape accesses the dimensions of the image array)
    brightImg = image.copy()
    for i in range(rows): 
        for j in range(columns):
            for k in range(colors):
                brightImg[i][j][k] += value
                if (brightImg[i][j][k] > 255): # Ensures that the RGB value does not exceed 255
                    brightImg[i][j][k] = 255
                if (brightImg[i][j][k] < 0): # Ensures that the RGB value does not go below 0
                    brightImg[i][j][k] = 0

    return brightImg

# Same principle from the change_brightness function regarding geting variables via image.shape and using a for loop
def change_contrast(image, value):
    rows, columns, colors = image.shape
    contrastImg = image.copy()
    factor = (259 * (value + 255)) / (255 * (259 - value)) # Formula taken from section 2.2.2 of project pdf
    for i in range(rows): 
        for j in range(columns):
            for k in range(colors):
                contrastImg[i][j][k] = factor * (image[i][j][k] - 128) + 128  # Uses formula from section 2.2.2
                if (contrastImg[i][j][k] > 255): # Ensures that the RGB value does not exceed 255
                    contrastImg[i][j][k] = 255
                elif (contrastImg[i][j][k] < 0): # Ensures that the RGB value does not go below 0
                    contrastImg[i][j][k] = 0
    return contrastImg

# Uses image.shape to obtain the values of rows, columns, colors so that they can be iterated via a for loop
def grayscale(image):
    rows, columns, colors = image.shape
    grayImg = image.copy()
    for i in range(rows):
        for j in range(columns):
            red = grayImg[i][j][0] # Red is the first color in "RGB", so array index is 0
            green = grayImg[i][j][1]  # Green is the second color in "RGB", so array index is 1
            blue = grayImg[i][j][2]  # Blue is the third color in "RGB", so array index is 2
            grayscaled = int(0.3 * red + 0.59 * green + 0.11 * blue) # Formula taken from section 2.2.4
            for k in range(colors):
                grayImg[i][j][k] = grayscaled # Replaces value
    return grayImg 

def blur_effect(image):
    blurredImg = image.copy()
    rows, columns, colors = image.shape
    # Need to add this adjustment since corner pixels do not have neighbors, leading to out of bound errors (used for edge detection/emboss as well)
    for i in range(1, rows- 1): 
        for j in range(1, columns - 1):
            for k in range(colors):
                # Formula taken from section 2.2.5
                blurredImg[i][j][k] = 0.0625 * image[i - 1][j - 1][k] + 0.125 * image[i - 1][j][k] + 0.0625 * image[i - 1][j + 1][k]  + 0.125 * image[i][j - 1][k]  + 0.25 * image[i][j][k] + 0.125 * image[i][j + 1][k]+ 0.0625 * image[i + 1][j - 1][k] + 0.125 * image[i + 1][j][k] + 0.0625 * image[i + 1][j + 1][k]
                if (blurredImg[i][j][k] > 255):  
                    blurredImg[i][j][k] = 255  # Ensures that the RGB value does not exceed 255 
                elif (blurredImg[i][j][k] < 0):
                    blurredImg[i][j][k] = 0 # Ensures that the RGB value does not go below 0
    return blurredImg 

#Same principle from blur_effect function, but the formula taken is from section 2.2.6
def edge_detection(image):
    newImg = image.copy()
    rows, columns, colors = image.shape
    for i in range(1, rows - 1): 
        for j in range(1, columns - 1):
            for k in range(colors):
                newImg[i][j][k] =  (-1 * image[i - 1][j - 1][k]) + (-1 * image[i - 1][j][k]) + (-1 * image[i - 1][j + 1][k] ) + (-1 * image[i][j - 1][k]) + (8 * image[i][j][k]) + (-1 * image[i][j + 1][k]) + (-1 * image[i + 1][j - 1][k]) + (-1 * image[i + 1][j][k]) + (-1 * image[i + 1][j + 1][k]) 
                newImg[i][j][k] += 128 # Reminder to add 128 after adding convolution kernel
                if (newImg[i][j][k] > 255): # "Clamps" RGB value
                    newImg[i][j][k] = 255
                elif (newImg[i][j][k] < 0):
                    newImg[i][j][k] = 0
    return newImg 

#Same principle from blur_effect function, but the formula taken is from section 2.2.7
def embossed(image):
    newImg = image.copy()
    rows, columns, colors = image.shape
    for i in range(1, rows - 1): 
        for j in range(1, columns - 1):
            for k in range(colors):
                newImg[i][j][k] =  (-1 * image[i - 1][j - 1][k]) + (-1 * image[i - 1][j][k]) + (-1 * image[i][j - 1][k]) + (image[i][j + 1][k]) + (image[i + 1][j][k]) + (image[i + 1][j + 1][k]) 
                newImg[i][j][k] += 128  # Reminder to add 128 after adding convolution kernel
                if (newImg[i][j][k] > 255):  
                    newImg[i][j][k] = 255
                elif (newImg[i][j][k] < 0):
                    newImg[i][j][k] = 0
    return newImg 

def rectangle_select(image, x, y):
    #Defining parameters of the function rectangle select
    rows = np.shape(image)[0]
    cols = np.shape(image)[1]

    x_i = x[0] # Initial x-coordinate of rectangle
    x_f = x[1] # Final x-coordinate of rectangle
    y_i = y[0] # Initial y-coordinate of rectangle
    y_f = y[1] # Final y-coordinate of rectangle
    

    rect = np.zeros((rows, cols))   
    rect[x_i : y_i + 1, x_f : y_f + 1] = 1 # Fix the range of rectangle with its diagonal from point (x_i, y_i) to (x_f, y_f)

    #Output the rectangle function
    print(rect) 
    return rect

def distance(image, pix1, pix2):
    #Finding the RGB values of the 2 pixels
    p1 = image[pix1[0], pix1[1]]
    p2 = image[pix2[0], pix2[1]]
    #Finding difference between the 3 colors
    deltaRed = p1[0] - p2[0]
    deltaGreen = p1[1] - p2[1]
    deltaBlue = p1[2] - p2[2]
    redAvg = (p1[0] + p2[0]) / 2
    #Using formula given in project document section 2.3.2
    return math.sqrt((2 + redAvg / 256) * (deltaRed ** 2) + 4 * (deltaGreen ** 2) + (2 + (255 - redAvg) / 256) * (deltaBlue ** 2))

def magic_wand_select(image, x, thres):
    #Finding the dimensions of the image
    row, column = np.shape(image)[:2]
    #Starting pixel is x
    stack = []
    stack.append(x)
    #Empty list to be filled up by visited pixels
    visitedList = []

    while len(stack) > 0:
        #Pop the current pixel from the stack and add it to visited list
        currentPix = stack.pop()
        visitedList.append(currentPix)
        #Find valid neighbours to add to the stack
        validNeighbors = findingValidNeighbors(image, currentPix, visitedList, thres,x)
        stack.extend(validNeighbors)
    #Create a mask from the list of pixels 
    return create_mask(visitedList, row, column)

#Function to find valid neighbouring pixels, based on color threshold
def findingValidNeighbors(image, currentPix, visitedList, thres,x):
    row, col,_ = image.shape
    #Up, down, left, right respectively
    neighborDirection = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    #List to store valid neighbors
    validNeighbors = []

    for direction in neighborDirection:
        nb = (currentPix[0] + direction[0], currentPix[1] + direction[1])
        #Checking if neighbor is valid or not (not out of bounds, within threshold, not visited)
        #Then adding them if they are valid
        if isValidNeighbor(nb, row, col, image, currentPix, thres, visitedList,x):
            validNeighbors.append(nb)
    return validNeighbors

#Function to check if neighbor is valid
def isValidNeighbor(nb, row, col, image, currentPix, thres, visitedList,x):
    #not out of bounds, within threshold, not visited
    return (0 <= nb[0] < row and 0 <= nb[1] < col and distance(image, nb, x) <= thres and nb not in visitedList)


#Function to create a mask from selected pixels
def create_mask(visitedList, row, col):
    #Empty mask filled with zeros
    msk = np.zeros((row, col), dtype=int)
    #Make pixels that are in visited list as 1 in the mask (instead of zero)
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

    # Changed from y-x to x-y
    print("Image size is",str(len(image[0])),"x",str(len(image)))

def applyMask(newImage, originalImage, mask):
    finalImage = originalImage.copy() # Creates a copy of the original image
    for i in range(len(originalImage)): # Iterates through the rows
        for j in range(len(originalImage[i])): # Iterates through each individual cell in the row
            # Checks whether or not to copy the pixel or not
            if (mask[i][j] == 1):
                finalImage[i][j] = newImage[i][j] 
            else:
                pass
    return finalImage

def menu():
    
    # Create "None" and boolean variables
    image = None
    newImg = None
    mask = None 
    newMask = None

    # Initial menu (with only "exit" and "load" options)
    while True:
        if image is None:
            # When user hasn't uploaded an image
            userSelect = input("What do you want to do ? \n e - exit \n l - load a picture \n \n Your choice: ")
        else: 
            # When user has uploaded an image
            userSelect = input("What do you want to do ? \n e - exit \n l - load a picture \n s - save the current picture \n 1 - adjust brightness \n 2 - adjust contrast \n 3 - apply grayscale \n 4 - apply blur \n 5 - edge detection \n 6 - embossed \n 7 - rectangle select \n 8 - magic wand select \n \n Your choice: ")
        
        # Done to exit the program
        if userSelect == "e":
            print("Thank you very much for using this picture editor. Have a nice day!")
            break
        elif userSelect == "l":
            while True:
                try:
                    # List of forbidden characters for file names (tested on a Microsoft, and those are the forbidden ones)
                    forbiddenChars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']  
                    while True:
                        filename = input("Enter the filename to load: ")
                        hasForbiddenChar = False
                        
                        # Check if any forbidden characters are in the input
                        for char in filename:
                            if char in forbiddenChars:
                                print(f"Filename contains forbidden character '{char}'. Please enter a valid filename.")
                                hasForbiddenChar = True
                                break  # Exit the for loop early if a forbidden character is found
                            else:
                                break
                        
                        # Ends loop if there are no forbidden characters
                        if not hasForbiddenChar:
                            break

                    # Continue with your code
                    start_time = time.time() 
                    image, mask = load_image(filename)
                    newImg = image.copy()
                    display_image(image, mask)
                    break
                except Exception as e:
                    print(f"An error occurred: {e}. Please enter a valid file name. You might have forgotten to input \".jpg\" or \".png\" or some other file format in your file name.")
            
            end_time = time.time()
            print(f"Image loaded in {end_time - start_time} seconds.")
        
        elif userSelect == "s" and image is not None:
            #Start timer
            start_time = time.time()
            forbiddenChars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']  
            while True:
                newFileName = input("Enter the name of your new file: ")
                hasForbiddenChar = False
                # Check if any forbidden characters are in the input (same logic as above)
                for char in newFileName:
                    if char in forbiddenChars:
                        print(f"Filename contains forbidden character '{char}'. Please enter a valid filename.")
                        hasForbiddenChar = True
                        break  # Exit the for loop early if a forbidden character is found
                
                # Ends loop if there are no forbidden characters
                if not hasForbiddenChar:
                    break

            # Choose format for someone to save it as (in case they forget to type the file type)
            while True:
                imgFormat = input("What would you like to save your format as? Click 1 for .jpg, Click 2 for .png, Click 3 for .gif! ")
                if imgFormat == "1":
                    imgFormat = ".jpg"
                    break
                elif imgFormat == "2":
                    imgFormat = ".png"
                    break
                elif imgFormat == "3":
                    imgFormat = ".gif"
                    break
                else:
                    print("Invalid input. Please try again!")
            
            # Calls the save function           
            save_image(newFileName + imgFormat, newImg)
            end_time = time.time()
            print(f"Image saved in {end_time - start_time} seconds.")
        
        # Note if userSelected == "1" and image is None, it tells the user to prompt something else
        elif userSelect == "1" and image is not None: 
            while True:
                # "Try block" to handle issues if the user accidentally puts in a float or string
                try:
                    rgbValue = int(input("Enter an input value to change the image brightness (has to be an integer): "))
                    # Checks if value is acceptable
                    if rgbValue < -250 or rgbValue > 250:
                        print("Please input an integer between -250 and 250. This exceeds the RGB maximum value (255) or minimum value (0).")
                        continue
                    break
                except ValueError:
                    print("Invalid input. Please enter an integer.") # In case user inputs float/string
            # Start function timer
            start_time = time.time()

            # Execute function
            modifiedImg = change_brightness(newImg, rgbValue)

            # Check if the new mask needs to be used (when 7/8 has been selected)
            if newMask is not None:
                newImg = applyMask(modifiedImg, newImg, newMask) 
            else:
                newImg = modifiedImg

            # Stops function timer
            end_time = time.time()
            print(f"Brightness adjusted in {end_time - start_time} seconds.")
            display_image(newImg, mask) # Displays the image desired
        
        # Note if userSelected == "2" and image is None, it tells the user to prompt something else
        elif userSelect == "2" and image is not None:
            while True:
                # "Try block" to handle issues if the user accidentally puts in a float or string
                try:
                    contrastValue = int(input("Enter an input value to change the image contrast (has to be an integer): "))
                    break 
                except ValueError:
                    print("Invalid input. Please enter an integer.")

            # Timer starts
            start_time = time.time()

            # Execute function
            modifiedImg = change_contrast(newImg, contrastValue)
            if newMask is not None:
                newImg = applyMask(modifiedImg, newImg, newMask) 
            else:
                newImg = modifiedImg
            end_time = time.time()
            print(f"Contrast adjusted in {end_time - start_time} seconds.")
            display_image(newImg, mask)
        
        # Note if userSelected == "3" and image is None, it tells the user to prompt something else
        elif userSelect == "3" and image is not None:
            start_time = time.time()

            # Execute function
            modifiedImg = grayscale(newImg)
            
            # Check if the new mask needs to be used (when 7/8 has been selected)
            if newMask is not None:
                newImg = applyMask(modifiedImg, newImg, newMask) 
            else:
                newImg = modifiedImg
            end_time = time.time()
            print(f"Grayscale applied in {end_time - start_time} seconds.")
            display_image(newImg, mask)
        
        # Note if userSelected == "4" and image is None, it tells the user to prompt something else
        elif userSelect == "4" and image is not None:
            start_time = time.time()

            # Execute function
            modifiedImg = blur_effect(newImg)

            # Check if the new mask needs to be used (when 7/8 has been selected)
            if newMask is not None:
                newImg = applyMask(modifiedImg, newImg, newMask)  
            else:
                newImg = modifiedImg
            end_time = time.time()
            print(f"Blur applied in {end_time - start_time} seconds.")
            display_image(newImg, mask)
        
        # Note if userSelected == "5" and image is None, it tells the user to prompt something else
        elif userSelect == "5" and image is not None:
            start_time = time.time()
            modifiedImg = edge_detection(newImg)

            # Check if the new mask needs to be used (when 7/8 has been selected)
            if newMask is not None:
                newImg = applyMask(modifiedImg, newImg, newMask) 
            else:
                newImg = modifiedImg
            end_time = time.time()
            print(f"Edge detection applied in {end_time - start_time} seconds.")
            display_image(newImg, mask)
        
        # Note if userSelected == "6" and image is None, it tells the user to prompt something else
        elif userSelect == "6" and image is not None:
            start_time = time.time()

            # Execute function
            modifiedImg = embossed(newImg)

            # Check if the new mask needs to be used (when 7/8 has been selected)
            if (newMask is not None):
                newImg = applyMask(modifiedImg, newImg, newMask)  
            else:
                newImg = modifiedImg
            end_time = time.time()
            print(f"Embossing applied in {end_time - start_time} seconds.")
            display_image(newImg, mask)
        
        # Note if userSelected == "7" and image is None, it tells the user to prompt something else
        elif userSelect == "7" and image is not None:
            while True:
                # Checks for x1, x2, y1, y2 values and ensures that they are valid
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
                    print("Invalid input. Please enter an integer value.") # Checks for error
            start_time = time.time()
            top = (y1, x1) # y1, x1 this order since it is row-column format
            bottom = (y2, x2) # y2, x2 this order since it is row-column format
            newMask = rectangle_select(image, top, bottom)
            end_time = time.time()
            print(f"Rectangle selected in {end_time - start_time} seconds.")
            display_image(newImg, newMask)

        
        elif userSelect == "8" and image is not None:
            while True:
                try:
                    # Checks for x, y, and threshold values and ensures that they are valid
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
            newMask = magic_wand_select(image, (yCoord, xCoord), thres) # Since row-col tuple, y and x coordinates switch
            end_time = time.time()
            print(f"Magic wand selection applied in {end_time - start_time} seconds.")
            display_image(newImg, newMask)
        
        else:
            print("Invalid choice. Please try again.")
            continue

if __name__ == "__main__":
    menu()






