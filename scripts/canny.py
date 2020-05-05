import numpy as np
import cv2
import itertools
import sys
import ntpath
from os import path

class CannyEdgeDetector:
    def __init__(self, 
                hyst_thresh_min, 
                hyst_thresh_max,
                gauss_size = 5, 
                gauss_stdv = 0.1):
        
        # Parameters for the first step of the algorithm
        self.gauss_size = gauss_size
        self.gauss_stdv = gauss_stdv

        # Parameters for the final step of the algorithm
        self.hyst_thresh_min = hyst_thresh_min
        self.hyst_thresh_max = hyst_thresh_max

    def apply(self, source_image):
        smoothed_image = self.smooth_image(source_image)
        G_mag, G_dir = self.calculate_intensity_gradient(smoothed_image)
        G_mag_nmx = self.non_max_suppression(G_mag, G_dir)
        return self.hysteresis_thresholding(G_mag_nmx)

    def smooth_image(self, source_image):
        """ 
            Input : source grayscale image
            First step of the algorithm : smooth image to reduce noise.
            We do this by applying a gaussian blur, given its parameters.
            Output : blurred source grayscale image
        """
        return cv2.GaussianBlur(source_image, (self.gauss_size, self.gauss_size), self.gauss_stdv)

    def calculate_intensity_gradient(self, smoothed_image):
        """
            Input : blurred source grayscale image
            Second step of the algorithm : calculate intensity gradient magnitude and direction
            We do this by calculating sobel X and sobel Y :
                - The magnitude is given by the approximation G_mag = |G_x|+|G_y|
                - The direction is given by G_dir = arctan((G_x) / (G_y)))
            Output: gradient magnitude and direction matrices
        """
            
        # Apply sobel filters in both directions x and y
        G_x = cv2.Sobel(smoothed_image, cv2.CV_64F, dx = 1, dy = 0, ksize = 3)
        G_y = cv2.Sobel(smoothed_image, cv2.CV_64F, dx = 0, dy = 1, ksize = 3)
            
        # Calculate gradient magnitude G_mag = |G_x|+|G_y|
        G_mag = cv2.addWeighted(cv2.convertScaleAbs(G_x), 1, cv2.convertScaleAbs(G_y), 1, 0, None)
            
        # Calculate gradient direction G_dir = arctan((sobel_y*I) / (sobel_x*I))), arctan2 returns between [pi, -pi]
        G_dir = np.arctan2(G_y, G_x)
            
        return G_mag, G_dir

    def non_max_suppression(self, G_mag, G_dir):
        """
            Input : gradient magnitudes and direction
            Third step of the algorithm : edge thinning on the gradient magnitude matrice
            For each pixel (except the border ones):
                - Look at its two neighbors in the direction of the gradient. We need to consider a range
                of angles that will define a direction for picking the neighbot pixels.
                - If its magnitude is maximal leave it unchanged, if not set it to 0
            Output : modified gradient magnitude matrix
        """
            
        h, w = G_mag.shape
        G_mag_nmx = G_mag.copy()

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                    
                # Get current pixel gradient direction (between pi and -pi)
                # We can consider the abs because for theta and -theta we will check the same neighbors
                grad_direction = np.abs(G_dir[i, j])

                # Left to right
                if 0 <= grad_direction < (0. + np.pi/8) or (np.pi - np.pi/8) < grad_direction <= np.pi:
                    n1 = G_mag_nmx[i, j-1]
                    n2 = G_mag_nmx[i, j+1]
                    
                # Bottom left to top right
                elif (np.pi/4 - np.pi/8) <= grad_direction < (np.pi/4 + np.pi/8):
                    n1 = G_mag_nmx[i+1, j-1]
                    n2 = G_mag_nmx[i-1, j+1]
                    
                # Bottom to top
                elif (np.pi/2 - np.pi/8) <= grad_direction < (np.pi/2 + np.pi/8):
                    n1 = G_mag_nmx[i+1, j]
                    n2 = G_mag_nmx[i-1, j]
                    
                # Bottom right to top left
                elif (3*np.pi/4 - np.pi/8) <= grad_direction < (3*np.pi/4 + np.pi/8):
                    n1 = G_mag_nmx[i+1, j+1]
                    n2 = G_mag_nmx[i-1, j-1]
                    
                if n1 >= G_mag_nmx[i, j] or  n2 >= G_mag_nmx[i, j]:
                    G_mag_nmx[i, j] = 0
            
        return G_mag_nmx

    def hysteresis_thresholding(self, G_mag_nmx, minVal = None, maxVal = None):
        """
            Input : gradient intensity matrix with non max suppression applied
            Final step of the algorithm to keep only strong edges (and edges connected to strong ones)
            Given min and max values of thresholding:
                - If an edge falls under the min, we discard it (setting it to 0)
                - If an edge falls above the max, we keep it (setting it to 255)
                - If an edge falls in between :
                    - If it is connected to a strong edge, we keep it (setting it to 255)
                    - Else we discard it
            Output: Thresholded gradient intensity matrix with non max suppression applied
        """
        if minVal is None:
            minVal = self.hyst_thresh_min
        if maxVal is None:
            maxVal = self.hyst_thresh_max

        h, w = G_mag_nmx.shape
        G_thresh = G_mag_nmx.copy()
        G_thresh[G_thresh < minVal] = 0
        G_thresh[G_thresh > maxVal] = 255
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if G_thresh[i, j] in [0, 255]:
                    continue
                else:
                    neighbors = np.array([G_thresh[i + offset[0], j + offset[1]] for offset in list(itertools.product([-1, 1, 0],[-1, 1, 0])) if (offset[0]!=0 or offset[1]!=0)])
                    G_thresh[i, j] = 0 if np.all(neighbors != 255) else 255
                    
        return G_thresh

def show_input_output(input_image, output_image, show_description):
    cv2.imshow(show_description, np.hstack((input_image, output_image)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_output(output_image, original_image_path, save_suffix):
    dirname = ntpath.dirname(original_image_path)
    filename = ntpath.basename(original_image_path).split('.')[0]
    extension = ntpath.basename(original_image_path).split('.')[1]

    result_path = ntpath.join(dirname, filename + save_suffix + "." + extension)
    cv2.imwrite(result_path, output_image)

def main():
    """
        Main function execution call:
            python canny.py <exec_mode> <path_to_image> <hyst_min> <hyst_max>
        (only thresholding params are usable via the command line)
        
        Exec_mode values :
            "show" (use open cv to display horizontally stacked the original image and its processed version)
            "save" (save processed version in the same directory as the original image by appending "_canny" to its name)
            "webcam" (use opencv canny implementation to apply it on webcam feed, <path_to_image> needs to be "None")
    """

    # Parse arguments
    if len(sys.argv) != 5:
        raise TypeError('Incorrect number of arguments, usage is : python canny.py <exec_mode> <path_to_image> <hyst_min> <hyst_max>')

    exec_mode = sys.argv[1]
    if exec_mode not in ["show", "save", "webcam"]:
        raise ValueError("Exec mode argument {} is not supported (show or save)".format(exec_mode))

    input_image_path = sys.argv[2]

    try:
        hyst_min = int(sys.argv[3])
        hyst_max = int(sys.argv[4])
    except:
        raise ValueError("One of the thresholding parameters is not an int : {}, {}".format(sys.argv[3], sys.argv[4]))

    if exec_mode == "webcam":
        if input_image_path != "None":
            raise ValueError("Please type None for the <path_to_image> argument in webcam exec_mode")
        
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canny_frame = cv2.Canny(frame, hyst_min, hyst_max)

            cv2.imshow('Canny edge detection',canny_frame)
            # Press Q to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        return

    # Run module on input_image_path
    if not path.exists(input_image_path):
        raise ValueError("Incorrect path for input image : {}".format(input_image_path))

    input_image = cv2.imread(input_image_path, 0)
    edge_detector = CannyEdgeDetector(hyst_min, hyst_max)
    output_image = edge_detector.apply(input_image)

    # Show or save result
    if exec_mode == "show":
        show_input_output(input_image, output_image, 'Original image (left), Canny edge detection (right)')
    elif exec_mode == "save":
        save_output(output_image, original_image_path, '_canny')

if __name__ == '__main__':
    main()