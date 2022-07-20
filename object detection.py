import cv2
import time
import imutils

cam = cv2.videocapture(1)
time.sleep(1)

fristframe = None
area = 500

while True:
    _,img = cam.read()
    text = "normal"
    img = imutils.resize(img,width=500)
    grayimg = cv2.cvtcolor(img,cv2.COLOR_BGR2GRAY)
    gaussianimg = cv2.GaussianBlur(grayimg,(21,21),0)
    if fristframe is None:
        fristframe = gaussianimg
        continue
    imgdiff = cv2.absdiff(fristframe , gaussianimg)
    three = cv2.threshold(imgdiff,25,255,cv2.THRESH_BINARY)[1]
    three = cv2.dilate(three,None,iterations=2)
    cnts = cv2.findcontours(three.copy(),cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourarea(c) < area:
            continue
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x + w , y + h),(0,255,0),2)
        text = "moving object dedction"
    print(text)
    cv2.putText(img,text, (10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
    cv2.imshow("camerafeed",img)
    key = cv2.waitKey(1)
    if key == ord("d"):
        break

cam.release()
cv2.destroyAllWindows()

    
                    
