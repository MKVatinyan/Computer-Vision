import numpy as np
import cv2
import sys

def keep_hue_in_range(image, hue_min, hue_max, sat_min, val_min):
    
    # If values don't make sense do nothing
    if hue_max <= hue_min or hue_max <= 0 or hue_max >= 180 or hue_min >= 179 or hue_min < -180:
        return image
    
    # Convert to useful color spaces
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # If min is negative, create two masks
    if (hue_min < 0):
        mask1 = cv2.inRange(image_hsv, (0, sat_min, val_min), (hue_max, 255, 255))
        mask2 = cv2.inRange(image_hsv, (180 + hue_min, sat_min, val_min), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2) // 255
    else:
        mask = cv2.inRange(image_hsv, (hue_min, sat_min, val_min), (hue_max, 255, 255)) // 255
    
    # Apply mask on image, move everything else to gray
    image_hsv[:, :, 0] *= mask
    image_hsv[:, :, 1] *= mask
    image_hsv[:, :, 2] *= mask
    
    # Use the grayscale image for all the rest of the zone
    mask_inverse = cv2.bitwise_not(mask) // 255
    image_hsv[:, :, 2] = image_gray * mask_inverse + image_hsv[:, :, 2] 

    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

def run_video_capture_hue_thresh(hmin_pos, hmax_pos, preset = None):
    
    if preset not in ["red", "blue", "green", None]:
        raise ValueError("Unknown preset given : {}".format(preset))
        
    w_name = "Hue thresholding (press q to exit)"

    cap = cv2.VideoCapture(0)
    cv2.namedWindow(w_name)
    cv2.createTrackbar("Sat min", w_name, 0, 254, lambda x : x)
    cv2.createTrackbar("Val min", w_name, 0, 254, lambda x : x)
    
    # Hard values, so the result is not that good, best way would be to adapt these
    # according to the saturation and value
    if preset == "red":
        hmin_pos = -8
        hmax_pos = 8
    elif preset == "blue":
        hmin_pos = 100
        hmax_pos = 140
    elif preset == "blue":
        hmin_pos = 30
        hmax_pos = 85

    while(True):

        ret, frame = cap.read()
        sat_pos = cv2.getTrackbarPos("Sat min", w_name)
        val_pos = cv2.getTrackbarPos("Val min", w_name)

        processed_frame = keep_hue_in_range(frame, hmin_pos, hmax_pos, sat_pos, val_pos)
        cv2.imshow(w_name, processed_frame)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
        Main function execution call:
            python huethresh.py <hue_min> <hue_max> (<preset>)

        This will launch the webcam, press Q to exit. 

        Hue min and max determine the thresholding applied on the colors. 
        - Hue min can be negative, so that we can correctly capture the "red" section : [-180, 179]
        - Hue max is in [0, 180]
        - If these conditions are not respected or if max is smaller then min, then nothing will happen
    """

    # Parse arguments
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        raise TypeError('Incorrect number of arguments, usage is : python huethresh.py <hue_min> <hue_max> (<preset>)')

    try:
        hue_min = int(sys.argv[1])
        hue_max = int(sys.argv[2])
    except:
        raise ValueError("One of the thresholding parameters is not an int : {}, {}".format(sys.argv[3], sys.argv[4]))

    if len(sys.argv) == 4:
        preset = sys.argv[3]
    else:
        preset = None

    run_video_capture_hue_thresh(hue_min, hue_max, preset)

if __name__ == '__main__':
    main()