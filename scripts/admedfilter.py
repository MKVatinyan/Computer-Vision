import numpy as np
import cv2
import sys
import ntpath

def adaptive_median_filter(image, w_max, w_0 = 3):
    """
        Adaptive median filtering (or classic median filtering if w_max = w_0)
        Input : BGR image (open cv default), w_max, w_0
        If the image is grayscale apply the algorithm, if it's color apply the algorithm on each channel
        Output : filtered BGR image
    """
    assert w_max % 2 == 1, "window size must be odd"

    if len(image.shape)==2:
        return adaptive_median_filter_single_channel(image, w_max, w_0)

    elif len(image.shape)==3:
        # We assume the image is opened with opencv so is BGR
        blue_channel = image[:, :, 0]
        green_channel = image[:, :, 1]
        red_channel = image[:, :, 2]

        # Apply filtering on each channel
        filtered_blue_channel = adaptive_median_filter_single_channel(blue_channel, w_max, w_0)
        filtered_green_channel = adaptive_median_filter_single_channel(green_channel, w_max, w_0)
        filtered_red_channel = adaptive_median_filter_single_channel(red_channel, w_max, w_0)

        return np.dstack((filtered_blue_channel, filtered_green_channel, filtered_red_channel))
    
    else:
        raise ValueError("Input dimension {} is not supported (expecting 3D RGB or 2D Gray)".format(image.shape))

def adaptive_median_filter_single_channel(image, w_max, w_0):
    """
        Adaptive median filtering (or classic median filtering if w_max = w_0)
        Input : single channel image (gray or a color channel), w_max, w_0
        1. Pad image using cv2.BORDER_REPLICATE to be able to have windows of size (w_max, w_max) on each pixels
            N.B.1 the usage of padding is a choice, other ways of handling borders are possible (not explored here)
            N.B.2 the usage of cv2.BORDER_REPLICATE is also an arbitrary choice, there are other padding methods (not explored here)
        2. Apply algorithm step on each pixel of the original image
        Output : filtered BGR image (padding removed)
    """
    # Pad image, the type of padding will influence the result on the borders
    border_size = w_max//2
    padded_image = cv2.copyMakeBorder(image, 
                                      border_size, 
                                      border_size, 
                                      border_size, 
                                      border_size, 
                                      cv2.BORDER_REPLICATE)

    # Iterate over each pixel of the original image
    p_im_h, p_im_w = padded_image.shape
    result = np.zeros((p_im_h, p_im_w), dtype = np.uint8)
    
    for i in range(border_size, p_im_h-border_size):
        for j in range(border_size, p_im_w-border_size):
            result[i, j] = adaptive_median_filter_step(padded_image, (i, j), w_max, w_0)
            
    return result[border_size:p_im_h-border_size, border_size:p_im_w-border_size]

def adaptive_median_filter_step(image, pix_location, w_max, w_size):
    """
        Adaptive median filtering step (or classic median filtering if w_max = w_0)
        Input : padded input image, pixel location, w_max, w_0
         Processing (flowchart of the algorithm : https://www.researchgate.net/figure/Flowchart-of-adaptive-median-filter_fig13_275066514) : 
            1. Get current pixel and consider a (w_size, w_size) window centered on it
            2. Calculate max, min and median pixel values of the window
            3. 
                If (w_size = w_max) return median pixel value

                If (min < median < max) and (min < current pixel value < max) return current pixel value
                If (min < median < max) and (min >= current pixel value or current pixel value >= max) return median pixel value
                If (min >= median or median >= max), increase w_size by 2 and repeat algorithm from 1.
            
        Output : pixel value on pixel location for the resulting image (current pixel value or median of a window of variable size)
    """
    i, j = pix_location
    current_pixel = image[i,j]

    # Init window centered on pixel (non square window on borders)
    h0 = i - w_size//2
    h1 = i + w_size//2 + 1
    w0 = j - w_size//2
    w1 = j + w_size//2 + 1
    
    # Extract window
    window = image[h0:h1, w0:w1]
    
    # Extract min, max and median
    min_val = np.min(window)
    max_val = np.max(window)
    med_val = np.median(window)

    if w_size == w_max:
        return med_val
            
    if min_val < med_val < max_val:
        return current_pixel if min_val < current_pixel < max_val else med_val
    else:
        return adaptive_median_filter_step(image, (i, j), w_max, w_size + 2)

def main():
    """
        Main function execution call:
            python admedfilter.py <exec_mode> <path_to_image> <w_max> <w_0>

        If w_max = w_0, then classic median filtering is applied with box size = w_max.
        If w_max != w_0, then adaptive median filtering is applied. If w_0 is empty, the default value 3 will be used.
        
        Exec_mode values :
            "show" (use open cv to display horizontally stacked the original image and its processed version)
            "save" (save processed version in the same directory as the original image by appending "_medfilter" to its name)
    """
    exec_mode = sys.argv[1]
    image_path = sys.argv[2]
    w_max = sys.argv[3]

    try:
        w_max = int(w_max)
    except:
        raise ValueError("w_max argument is not an int : {}".format(w_max))

    if len(sys.argv) == 4:
        w_0 = 3
    elif len(sys.argv) == 5:
        try:
            w_0 = int(sys.argv[4])
        except:
            raise ValueError("w_0 argument is not an int : {}".format(w_0))
    else:
        raise TypeError('Incorrect number of arguments, usage is : python admedfilter.py <exec_mode> <path_to_image> <w_max> (<w_0>)')

    image = cv2.imread(image_path, 0)
    filtered_image = adaptive_median_filter(image, w_max, w_0)
    
    if exec_mode == "show":
        
        cv2.imshow('Original image (left), Filtered image (right)', np.hstack((image, filtered_image)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif exec_mode == "save":
        dirname = ntpath.dirname(image_path)
        filename = ntpath.basename(image_path).split('.')[0]
        extension = ntpath.basename(image_path).split('.')[1]

        filtered_image_path = ntpath.join(dirname, filename + "_medfilter." + extension)
        cv2.imwrite(filtered_image_path, filtered_image)

    else: 
        raise ValueError("Exec mode argument {} is not supported (show or save)".format(exec_mode))

if __name__ == '__main__':
    main()