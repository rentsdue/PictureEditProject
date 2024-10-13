import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def change_brightness(image, value):
    return np.array([]) # to be removed when filling this function
  
def change_contrast(image, value):
    return np.array([]) # to be removed when filling this function

def grayscale(image):
    return np.array([]) # to be removed when filling this function

def blur_effect(image):
    return np.array([]) # to be removed when filling this function

def edge_detection(image):
    return np.array([]) # to be removed when filling this function

def embossed(image):
    return np.array([]) # to be removed when filling this function

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
    try:
        img = mpimg.imread(filename)
        if len(img[0][0]) == 4:  # if png file
            img = np.delete(img, 3, 2)
        if type(img[0][0][0]) == np.float32:  # if stored as float in [0,..,1] instead of integers in [0,..,255]
            img = img * 255
            img = img.astype(np.uint8)
        mask = np.ones((len(img), len(img[0])))  # create a mask full of "1" of the same size of the loaded image
        img = img.astype(np.int32)
        return img, mask, True  
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None, False 

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
    imageLoaded = False

    while True:
        if not imageLoaded:
            userSelect = input("What do you want to do ? \n e - exit \n l - load a picture \n \n Your choice: ")
        else: 
            userSelect = input("What do you want to do ? \n e - exit \n l - load a picture \n s - save the current picture \n 1 - adjust brightness \n 2 - adjust contrast \n 3 - apply grayscale \n 4 - apply blur \n 5 - edge detection \n 6 - embossed \n 7 - rectangle select \n 8 - magic wand select \n \n Your choice: ")

        if userSelect.lower() == "e":
            break
        elif userSelect.lower() == "l":
            filename = input("Enter the filename to load: ")
            image, mask, imageLoaded = load_image(filename)
            load_image()
        elif userSelect.lower() == "s" and imageLoaded:
            save_image()
        elif userSelect.lower() == "1" and imageLoaded:
            change_brightness()
        elif userSelect.lower() == "2" and imageLoaded:
            change_contrast()
        elif userSelect.lower() == "3" and imageLoaded:
            grayscale()
        elif userSelect.lower() == "4" and imageLoaded:
            blur_effect()
        elif userSelect.lower() == "5" and imageLoaded:
            edge_detection()
        elif userSelect.lower() == "6" and imageLoaded:
            embossed()
        elif userSelect.lower() == "7" and imageLoaded:
            rectangle_select()
        elif userSelect.lower() == "8" and imageLoaded:
            magic_wand_select()
        else:
            print("Invalid Input Detected. Please Type a Valid Number or Letter.")
    
if __name__ == "__main__":
    menu()






