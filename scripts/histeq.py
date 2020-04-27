import numpy as np
import cv2
import sys
import ntpath

def equalize_image_histogram(image):
    """
        Basic image histogram equalization function.
        Input : BGR image (open cv default)
        Processing : 
            1. Switch from BGR to HSV and get value/intensity map of image
            2. Equalize the histogram of values, leaving hue and saturation maps untouched
                Source : https://en.wikipedia.org/wiki/Histogram_equalization, Implementation section
            3. Switch the resulting HSV image to BGR to display or save
        Output : BGR image with equalized histogram
    """
    if len(image.shape) == 3:

        # Image is assumed to be BGR (open cv default)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Apply this function recursively on a single intensity channel
        image_hsv[:, :, 2] = equalize_image_histogram(image_hsv[:, :, 2])

        # Switch back to RGB
        return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    elif len(image.shape) == 2:

        # Image is assumed to be grayscale / value channel
        # Number of possible intensity values
        L = 256
        
        # Calculate the normalized histogram
        hist = cv2.calcHist([image], [0], None, [L], [0,L])
        hist /= hist.max()

        # Calculate the normalized cumulative distribution function
        cdf  = np.cumsum(hist)
        cdf /= cdf.max()
    
        # Apply formula to get new intensity value, keep float for precision
        unscaled_heq_image = ((L-1) * cdf[image])
    
        # Scale back to be in [0, 255], we don't change pixels which had 0 intensity in the original
        hist_min = np.min(hist[np.nonzero(hist)])
        scaled_heq_image = (unscaled_heq_image - hist_min)/(hist.max() - hist_min)
    
        return scaled_heq_image.astype(np.uint8)

    else:
        raise ValueError("Input dimension {} is not supported (expecting 3D RGB or 2D Gray)".format(image.shape))

def main():
    """
        Main function execution call:
            python histeq.py <path_to_image> <exec_mode>
        Exec_mode values :
            "show" (use open cv to display horizontally stacked the original image and its processed version)
            "save" (save processed version in the same directory as the original image by appending "_eqnorm" to its name)
    """
    image_path = sys.argv[1]
    exec_mode = sys.argv[2]

    image = cv2.imread(image_path, 1)
    eq_norm_image = equalize_image_histogram(image)

    if exec_mode == "show":
        
        cv2.imshow('Original image (left), Equalized histogram image (right)', np.hstack((image, eq_norm_image)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif exec_mode == "save":
        dirname = ntpath.dirname(image_path)
        filename = ntpath.basename(image_path).split('.')[0]
        extension = ntpath.basename(image_path).split('.')[1]

        eq_norm_image_path = ntpath.join(dirname, filename + "_eqnorm." + extension)
        cv2.imwrite(eq_norm_image_path, eq_norm_image)

    else: 
        raise ValueError("Exec mode argument {} is not supported (show or save)".format(exec_mode))

if __name__ == '__main__':
    main()