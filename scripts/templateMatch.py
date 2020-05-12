import numpy as np
import cv2
import sys
import ntpath
from os import path

# Sources
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
# https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

def match_template(image, template, mode="single", thresh=None, max_rect=10):
    """
        Template matching:
            - mode = single : will look for the maximal ressemblence to match one patch
            - mode = multiple : will use thresh to match up to max_rect patches (not working that well)

        Matches will be drawn on the image as rectangles of the same size as the selected template.
        Since it's a direct pixel comparison it is not scale invariant. It will work best if the template
        pixels seem unique with respect to the background.
    """
    th, tw, _ = template.shape
    matching = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    
    if mode=="single":
        # Usage of minmax (cf. tutorial on website)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matching)
        top_left = max_loc
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        cv2.rectangle(image, top_left, bottom_right, (0,0,255), 2)

    elif mode=="multiple" and thresh is not None:
        # Usage of thresholding (cf. tutorial on website)
        loc = np.where( matching >= thresh)
        rect_count = 0
        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + tw, pt[1] + th), (0,0,255), 2)
            rect_count+=1
            if rect_count > max_rect:
                break
        print(rect_count)

    else:
        raise ValueError("Unknown mode : {}".format(mode))

    return image

def main():
    """
        Main function execution call:
            python templateMatch.py (<matching_mode> <thresh> <nmax_matches>)

        This will launch the webcam: 
            - Press p to pause video
            - While on pause, use the mouse to select a rectangle area on the frame
            - Once satisfied, press enter to validate the area as template and initiate the matching
            - Press p again to repeat the process
            - Press q to quit 
    """

    if len(sys.argv) != 2 and len(sys.argv) != 4:
        raise TypeError('Incorrect number of arguments, usage is : python templateMatch.py (<matching_mode> <nmax_matches>)')

    if len(sys.argv) == 4:
        matching_mode = sys.argv[1]
        try:
            thresh = float(sys.argv[2])
            nmax_matches = int(sys.argv[3])
        except:
            raise ValueError("One of the last two parameters has an incorrect format (first must be float, second must be int) : {}, {}".format(sys.argv[2], sys.argv[3]))
    else:
        matching_mode = "single"
        thresh = None
        nmax_matches = None

    # Define global variables
    global p1, template
    
    # Init global variables
    p1 = None
    template = None

    # Init flags
    template_validated = False
    is_recording = True

    # Define mouse callback function
    def click_and_select(event, x, y, flags, frame):

        global p1, template

        if event == cv2.EVENT_LBUTTONDOWN:
            p1 = (x, y)
            #cv2.imshow("frame", frame)

        elif event == cv2.EVENT_LBUTTONUP:
            p2 = (x, y)
            rect_frame = frame.copy()
            cv2.rectangle(rect_frame, p1, p2, (0,0,255), 2)
            cv2.imshow("frame", rect_frame)
            template = frame[p1[1]:p2[1], p1[0]:p2[0]]

    # Start video feed
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    out = cv2.VideoWriter("C:/Users/mkrva/Repositories/Computer-Vision/test.mp4", 
                            cv2.VideoWriter_fourcc(*'XVID'), 
                            25, 
                            (640, 480))
    while(True):

        ret, frame = cap.read()

        # When in pause mode, use mouse to select template, press enter to validate
        if not is_recording:
            cv2.setMouseCallback("frame", click_and_select, frame)
            while not template_validated:
                cv2.waitKey(1)
                if cv2.waitKey(1) & 0xFF == ord('\r'):
                    print("Template validated, shape :", template.shape)
                    template_validated = True
            is_recording = True
            cv2.setMouseCallback("frame", lambda *args : None)

        # When in video mode, press p to reset the template and pause the video
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print("Pausing video...")
            is_recording = False
            template_validated = False
            continue

        # Match template of the frames of the video
        frame = match_template(frame, template, matching_mode, thresh, nmax_matches) if template is not None else frame

        # Resume video feed
        if is_recording:
            cv2.imshow("frame", frame)
            out.write(frame)
            
        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break
        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()