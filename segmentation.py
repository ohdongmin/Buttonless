import cv2
import numpy as np

# define the lower and upper bounds of the color to be segmented
lower_color = np.array([0, 80, 80])
upper_color = np.array([10, 255, 255])

# initialize the camera
cap = cv2.VideoCapture(0)

# set the resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# define the quadrants of the screen
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
quadrants = [
    (0, 0, width//2, height//2), # top-left quadrant
    (width//2, 0, width, height//2), # top-right quadrant
    (0, height//2, width//2, height), # bottom-left quadrant
    (width//2, height//2, width, height) # bottom-right quadrant
]

while True:
    # capture a frame
    ret, frame = cap.read()

    # apply a Gaussian blur to the frame to reduce noise
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    # convert the frame to the HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # threshold the frame to extract the color to be segmented
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # find the contours in the thresholded image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # initialize a variable to store the detected quadrant
    detected_quadrant = None

    # iterate over the contours and check which quadrant they belong to
    for contour in contours:
        # calculate the centroid of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # check which quadrant the centroid belongs to
        for i, (x, y, w, h) in enumerate(quadrants):
            if x <= cx <= x+w and y <= cy <= y+h:
                detected_quadrant = i
                break

        # draw the contour and centroid on the frame
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # display the frame and detected quadrant
    for i, (x, y, w, h) in enumerate(quadrants):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'Q{i+1}', (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Detected quadrant: {detected_quadrant}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
# show the frame
    cv2.imshow('frame', frame)
    cv2.imshow("Mask", mask)


# check if the user wants to quit
    if cv2.waitKey(1) == ord('q'):
        break
