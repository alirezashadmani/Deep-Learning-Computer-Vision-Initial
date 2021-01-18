import cv2

def mouse_event(event,x,y,flag,param):
    if (event==cv2.EVENT_LBUTTONDOWN):
        points.append((x,y))
        if len(points)==2:
            cv2.rectangle(image,points[0],points[1],
                          (0,255,0),3)
            crop_image=image[points[-2][1]:points[-1][1],points[-1][-2][0]:points[-1][0]]
            cv2.imshow("frame1",crop_image)
    if (event==cv2.EVENT_MOUSEMOVE):
        points.append((x,y))
        if(len(points)==2):
            cv2.line(image,points[-2],points[-1](56,255,100),5)
        if (event==cv2.EVENT_RBUTTONMOUSE):
            blue=image[y,x,0]
            green=image[y,x,1]
            red=image[y,x,2]
            str1=str(blue)+","+str(green)+","+str(red)
            cv2.putText(image,str1,(x,y),
                        cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),5)
        image1=np.zeros((512,512,3),np.uint8)
        image1[:]=[blue,green,red]
        cv2.imshow("frame1",image1)
    
    cv2.imshow("Frame",image)

image=cv2.imread("Lena1.jpg")
points=[]
cv2.imshow("Frame",image)
cv2.setMouseCallback("Frame",mouse_event)

cv2.waitKey(0)
cv2.destroyAllWindow()
