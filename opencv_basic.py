
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import time

# img_file = "D:\ditto.jpg"
# img= cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
# cv2.imshow('Ditto',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

#2-4
# video_file = "D:\ditto_mv.mp4"
# cap = cv2.VideoCapture(video_file)
# if cap.isOpened():
#     while True:
#         ret, img = cap.read()
#         if ret:
#             cv2.imshow(video_file, img)
#             cv2.waitKey(3)
#         else:
#             break
# else:
#     print('no')
    
# cap.release()
# cv2.destroyAllWindows()

# video_file = "D:\ditto_short.mp4"
# cap = cv2.VideoCapture(video_file)
# if cap.isOpened():
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     delay = int(1000/fps)
#     print("FPS: %f, Delay: %dms" %(fps,delay))
#     while True:
#         ret, img = cap.read()
#         if ret:
#             cv2.imshow(video_file, img)
#             cv2.waitKey(delay)
#         else:
#             break
# else:
#     print('no')
    
# cap.release()
# cv2.destroyAllWindows()

#웹캠 테스트 
# cap = cv2.VideoCapture(0)
# if cap.isOpened():
#     while True:
#         ret, img = cap.read()
#         if ret:
#             cv2.imshow('camera',img)
#             if cv2.waitKey(1) != -1:
#                 break
# else:
#     print("no camera")
# cap.release()
# cv2.destroyAllWindows()

#창 관리
# file_path = "D:\ditto.jpg"
# img = cv2.imread(file_path, cv2.IMREAD_COLOR)
# img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

# cv2.namedWindow('origin', cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow('gray', cv2.WINDOW_NORMAL)

# cv2.imshow('origin', img)
# cv2.imshow('gray', img_gray)

# cv2.moveWindow('origin', 0, 0)
# cv2.moveWindow('gray', 100, 100)

# cv2.waitKey(0)
# cv2.resizeWindow('origin', 200,200)
# cv2.resizeWindow('gray', 100,100)

# cv2.waitKey(0)
# cv2.destroyWindow("gray")

# cv2.waitKey(0)
# cv2.destroyAllWindows()

#키보드 이벤트 처리
# img_file = "D:\ditto.jpg"
# img = cv2.imread(img_file)
# title = 'ditto'
# x,y = 100,100

# while 1:
#     cv2.imshow(title, img)
#     cv2.moveWindow(title,x,y)
#     key =cv2.waitKey(0) & 0xFF
#     print(key, chr(key))
#     if key == ord('h'):
#         x-=10
#     elif key == ord('j'):
#         y+= 10
#     elif key == ord('q') or key == 27:
#         break
#         cv2.destroyAllWindows()
#     cv2.moveWindow(title,x,y)
    

#마우스 이벤트 처리
# title = 'mouse event'
# img = cv2.imread('D:\ditto.jpg')
# cv2.imshow(title,img)


# def onMouse(event, x, y, flags, param):
#     print(event,x,y)
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_SHIFTKEY:
#             cv2.circle(img, (x,y), 30, (100,100,100), -1)
#             cv2.imshow(title, img)
#         elif flags & cv2.EVENT_FLAG_CTRLKEY :
#             cv2.circle(img, (x,y), 30, (0,100,0), -1)
#             cv2.imshow(title, img)

# cv2.setMouseCallback(title, onMouse)

# while True:
#     if cv2.waitKey(0) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

#트랙바 이미지 색 조정
# win_name = 'Trackbar'

# img = cv2.imread('D:\ditto.jpg')
# cv2.imshow(win_name,img)

# def onChange(x):
#     print(x)
#     r = cv2.getTrackbarPos('R',win_name)
#     g = cv2.getTrackbarPos('G',win_name)
#     b = cv2.getTrackbarPos('B',win_name)
#     print(r,g,b)
#     img[:] = [b,g,r]
#     cv2.imshow(win_name, img)

# cv2.createTrackbar('R', win_name, 155, 255, onChange)
# cv2.createTrackbar('G', win_name, 0, 255, onChange)
# cv2.createTrackbar('B', win_name, 255, 255, onChange)

# while True:
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

#matplotlib
# img = cv2.imread('D:\ditto.jpg')
# plt.imshow(img[:,:,::-1])
# plt.xticks([])
# plt.yticks([])
# plt.show()

#4-1 roi 지정, #4-2 관심영역 복제 및 새 창 띄우기
# img = cv2.imread('D:\ditto.jpg')
# title = 'mouse event'

# x = 153; y = 101; w= 50; h =50
# roi = img[y:y+h, x:x+w]
# img2 = roi.copy()
# img[y:y+h, x+w:x+w+w] = roi
# cv2.rectangle(roi, (0,0), (h-1,2*w-1),(0,255,0))
# cv2.imshow(title,img)
# cv2.imshow('roi',img2)

# def onMouse(event, x, y, flags, param):
#     print(event,x,y)
    
# cv2.setMouseCallback(title, onMouse)

# while True:
#     if cv2.waitKey(0) & 0xFF == 27:
#         break


#4-3 마우스로 관심영역 지정(roi_crop_mouse.py)
# isDragging = False
# x0, y0, w, h = -1,-1,-1,-1
# blue, red = (255,0,0), (0,0,255)

# def onMouse(event, x, y, flags, param):
#     global isDragging, x0, y0, img
#     if event == cv2.EVENT_LBUTTONDOWN:
#         isDragging = True
#         x0 = x
#         y0 = y
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if isDragging:
#             img_draw = img.copy()
#             cv2.rectangle(img_draw, (x0,y0), (x,y),blue, 2)
#             cv2.imshow('img',img_draw)
#     elif event == cv2.EVENT_LBUTTONUP:
#         if isDragging:
#             isDragging = False
#             w = x-x0
#             h = y - y0
#             if x>0 and h>0 :
#                 img_draw = img.copy()
#                 cv2.rectangle(img_draw, (x0,y0), (x,y),red, 2)
#                 cv2.imshow('img', img_draw)
#                 roi = img[y0:y0+h, x0:x0+w]
#                 cv2.imshow('cropped', roi)
#                 cv2.moveWindow('cropped',0,0)
#             else:
#                 cv2.imshow('img',img)
#                 print("다시 드래그해라")

# img = cv2.imread('D:\ditto.jpg')
# cv2.imshow('img',img)
# cv2.setMouseCallback('img',onMouse)
# cv2.waitKey()
# cv2.destroyAllWindows()

#마우스로 ROI 추출하기
# img = cv2.imread('D:\ditto.jpg')

# x,y,w,h = cv2.selectROI('img', img, True) # 스페이스나 엔터치면 추출
# if w and h:
#     roi = img[y:y+h, x:x+w]
#     cv2.imshow('cropped',roi)
#     cv2.moveWindow('cropped',0,0)
    
# cv2.imshow('img',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

#영상 나오는지 확인
# cap = cv2.VideoCapture(0)
# if cap.isOpened():
#     while True:
#         ret, frame = cap.read()
#         if ret:
#             cv2.imshow('camera',frame)
#             if cv2.waitKey(1) != -1:
#                 break

# cap.release()       
# cv2.destroyAllWindows()



# # 사진 200장 찍기
# # initialize the camera
# cap = cv2.VideoCapture(0)

# # set the resolution to 640x480
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# # capture 200 images at a rate of one per second
# for i in range(10):
#     # capture a frame
#     ret, frame = cap.read()
#     cv2.imshow('camera',frame)
#     # save the frame as a JPEG file
#     filename = f'test_{i+1}.jpg'
#     cv2.imwrite(filename, frame)
    
#     # wait for one second before capturing the next frame
#     time.sleep(1)
#     if cv2.waitKey(1) == ord('q'):
#         break
# # release the camera
# cap.release()
# cv2.destroyAllWindows()

#차영상 test -> 실패 조명 때문에 완전히 다름

# img1 = cv2.imread('D:\photo_none.jpg')
# img2 = cv2.imread('D:\photo_2.jpg')
# img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# diff = cv2.absdiff(img1_gray,img2_gray)
# _, diff = cv2.threshold(diff,1,255,cv2.THRESH_BINARY)
# diff_red = cv2.cvtColor(diff,cv2.COLOR_GRAY2BGR)
# diff_red[:,:,2] = 0

# spot = cv2.bitwise_xor(img2, diff_red)

# cv2.imshow('spot',spot)
# cv2.imshow('diff',diff)
# cv2.waitKey()
# cv2.destroyAllWindows()

thresh = 25
max_diff =5

a,b,c = None,None, None
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if cap.isOpened():
    ret, a = cap.read()
    ret, b = cap.read()

    while ret:
        ret, c = cap.read()
        draw = c.copy()
        if not ret:
            break

        a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        c_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

        diff1 = cv2.absdiff(a_gray, b_gray)
        diff2 = cv2.absdiff(b_gray, c_gray)

        ret, diff1_t = cv2.threshold(diff1, thresh, 255, cv2.THRESH_BINARY)
        ret, diff2_t = cv2.threshold(diff2, thresh, 255, cv2.THRESH_BINARY)
        
        diff = cv2.bitwise_and(diff1_t, diff2_t)

        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k)

        diff_cnt = cv2.countNonZero(diff)
        if diff_cnt > max_diff:
            nzero = np.nonzero(diff)
            cv2.rectangle(draw, (min(nzero[1]), min(nzero[0])),\
                                 (max(nzero[1]), max(nzero[0])),(0,255,0),2)
            cv2.putText(draw, "Motion Detected", (10,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255))

        stacked = np.hstack((draw, cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('motion sensor', stacked)

        a=b
        b=c

        if cv2.waitKey(1) & 0XFF == 27:
            break
    