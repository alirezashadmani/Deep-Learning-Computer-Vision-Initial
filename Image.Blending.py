import cv2
import numpy as np

apple=cv2.imread("apple.jpg")
orange=cv2.imread("orange.jpg")

apple_orange=np.hstack((apple[:,:256],orange[:,256:]))

lp_apple=[]
lp_orange=[]

from pyramid import laplacian
lp_apple=laplacian(apple)
lp_orange=laplacian(orange)


stack=[]
for apple_lap,orange_lap in zip(lp_apple, lp_orange):
    col, row, ch=apple_lap.shape
    lap=np.hstack((apple_lap[:,:int(col/2)],orange_lap[:,int(col/2):]))
    stack.append(lap)


recons=stack[0]
for i in range(1,6):
    recons=cv2.pyrUp(recons)
    recons=cv2.add(stack[i],recons)


cv2.imshow("hstack",apple_orange)
cv2.imshow("apple_orange",recons)
cv2.waitKey()
cv2.destroyAllWindows()
