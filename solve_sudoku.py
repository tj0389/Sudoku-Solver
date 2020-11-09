import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt 
import helper        ### helper is a python file in the folder in which all function are made previously


heightImg = 450
widthImg = 450

img=cv2.imread("test2.jpg")
img = cv2.resize(img, (widthImg, heightImg))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img_threshold = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,10)


img_contour=imgBlank.copy()
biggest,maxarea=helper.get_contour(img_threshold,img_contour)


imgwarpcolored=imgBlank.copy()
if biggest.size!=0:
    biggest=helper.reorder(biggest)
    pts1=np.float32(biggest)
    pts2=np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    imgwarpcolored=cv2.warpPerspective(img,matrix,(widthImg,heightImg))
imgwarpcolored_gray = cv2.cvtColor(imgwarpcolored,cv2.COLOR_RGB2GRAY)
imgwarpcolored_new = cv2.adaptiveThreshold(imgwarpcolored_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,10)
imgwarpcolored_new=cv2.bitwise_not(imgwarpcolored_new)
model=load_model("best_model_2.h5")
result=helper.getPredection(imgwarpcolored_new,model)
matrix=np.array(result)
matrix=matrix.reshape(9,9)
print(matrix)



imgsolveddigits=imgBlank.copy()
imgdetecteddigit=imgBlank.copy()
imgdetecteddigit=helper.displayNumbers(imgdetecteddigit,result,(0,0,153))
imgdetecteddigit=helper.addGrid(imgdetecteddigit)
numbers=np.asarray(result)
posArray=np.where(numbers>0,0,1)
matrix=np.array(posArray)
matrix=matrix.reshape(9,9)
print(matrix)



board = np.array_split(numbers,9)
print(board)
try:
    helper.solve(board)
except:
    pass
print(board)
flatList = []
for sublist in board:
    for item in sublist:
        flatList.append(item)
solvedNumbers =flatList*posArray
imgsolveddigits=imgBlank.copy()
imgsolveddigits= helper.displayNumbers(imgsolveddigits,solvedNumbers,(255,255,102))
imgsolveddigits_new=imgsolveddigits.copy()
imgsolveddigits=helper.addGrid(imgsolveddigits)




pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
try:
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgsolveddigits_new, matrix, (widthImg, heightImg))
    imgInvWarpColored = cv2.bitwise_not(imgInvWarpColored)
    imgInvWarpColored = cv2.bitwise_and(imgInvWarpColored,img)
except:
    print("Image is of Bad quality")
    imgInvWarpColored = img.copy()



img_stack=helper.stackImages(([img,img_threshold,imgwarpcolored_new],
                           [imgdetecteddigit,imgsolveddigits,imgInvWarpColored]),0.5)
cv2.imshow('Image',img_stack)
cv2.waitKey(0)
cv2.destroyAllWindows()