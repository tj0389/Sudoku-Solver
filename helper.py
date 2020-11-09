import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt 

def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
#             else:
#                 cv2.putText(img, str(numbers[(y*9)+x]),
#                                (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
#                             2, color, 2, cv2.LINE_AA)
    return img



def addGrid(img,color=(255,255,255)):
    factor=img.shape[0]//9
    img_new=img.copy()
    for i in range(10):
        cv2.line(img_new,(0,factor*i),(img_new.shape[1],factor*i),color,2,2)
        cv2.line(img_new,(factor*i,0),(factor*i,img_new.shape[0]),color,2,2)
    return img_new


def getPredection(image,model):
    result = []
    factor=image.shape[0]//9
    for i in range(9):
        for j in range(9):
            part = image[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            part = cv2.resize(part,(28,28))
            part=part[3:25,3:25]
            part = cv2.resize(part,(28,28))
            cv2.imwrite("Images/{}_{}.jpg".format(i,j),part)
            img = part.astype('float32')
            img=img/255.0
            img = img.reshape(1, 28, 28, 1)
            ## GET PREDICTION
            predictions = model.predict(img)
            classIndex=np.argmax(predictions)
            probabilityValue = np.amax(predictions)
            ## SAVE TO RESULT
            if probabilityValue > 0.6:
                result.append(classIndex)
            else:
                result.append(0)
    return result

def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i

            if solve(bo):
                return True

            bo[row][col] = 0

    return False


def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False

    return True


def print_board(bo):
    for i in range(len(bo)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")

        for j in range(len(bo[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(bo[i][j])
            else:
                print(str(bo[i][j]) + " ", end="")


def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col

    return None


def get_contour(img,target_image):
    contours,heiracy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    biggest=np.array([])
    maxArea=0
    for cnt in contours:
        area=cv2.contourArea(cnt)
        if (area>5000):
            #cv2.drawContours(img_contour,cnt,-1,(255,0,0),4)
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.099*peri,True)      ## return coordinates of points in contour 
            #print(len(approx))
            if area>maxArea and len(approx)==4:
                biggest=approx
                maxArea=area
        cv2.drawContours(target_image,biggest,-1,(255,0,255),20)
    return biggest,maxArea
