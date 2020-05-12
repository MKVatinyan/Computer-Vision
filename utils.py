import numpy as np
import cv2
import matplotlib.pyplot as plt
import ntpath

def run_video_capture(w_name = "Video Capture",
                        screenshot_directory = None,
                        process_frame = None):

    cap = cv2.VideoCapture(0)
    w_name += " (press q to quit, space to take screenshot if directory specified)"
    cv2.namedWindow(w_name)
    screenshot_idx = 0

    while(True):

        ret, frame = cap.read()
        if process_frame is not None:
            frame = process_frame(frame)
        
        cv2.imshow(w_name, frame)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord(' ') and screenshot_directory is not None:
            path = ntpath.join(screenshot_directory, "screenshot_{}.jpg".format(screenshot_idx))
            print("saving to {}".format(path))
            cv2.imwrite(path, frame)
            continue

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def resize_image(image, factor, interpolation):
    
    assert factor != 0.

    image_h = image.shape[0]
    image_w = image.shape[1]
    
    new_image_h = int(image_h*factor)
    new_image_w = int(image_w*factor)
    
    return cv2.resize(image, (new_image_w, new_image_h), interpolation = interpolation)

def shift_rgb_values(input_color_rgb, show = False, red = 0, green = 0, blue = 0):
    
    # Switch to uint16 to be able to go above 256
    shifted_img_color_rgb = input_color_rgb.copy().astype(np.uint16)
    
    # We use clip and not modulo to avoid cycling
    shifted_img_color_rgb[:,:,0] = np.clip(shifted_img_color_rgb[:,:,0] + red, 0, 255)
    shifted_img_color_rgb[:,:,1] = np.clip(shifted_img_color_rgb[:,:,1] + green, 0, 255)
    shifted_img_color_rgb[:,:,2] = np.clip(shifted_img_color_rgb[:,:,2] + blue, 0, 255)
    
    # Switch back to uint8 since with clip we went back below 256
    shifted_img_color_rgb = shifted_img_color_rgb.astype(np.uint8)

    # Show image
    if show:
        plt.imshow(shifted_img_color_rgb, interpolation = 'bicubic')
        plt.axis('off')
        plt.show()

        return
    else:
        return shifted_img_color_rgb

def shift_hsv_values(input_color_hsv, show = False, hue = 0, saturation = 0, value = 0):
    
    # Switch to uint16 to be able to go above 256
    shifted_img_color_hsv = input_color_hsv.copy().astype(np.uint16)
    
    # We use clip and not modulo to avoid cycling
    shifted_img_color_hsv[:,:,0] = np.clip(shifted_img_color_hsv[:,:,0] + hue, 0, 179)
    shifted_img_color_hsv[:,:,1] = np.clip(shifted_img_color_hsv[:,:,1] + saturation, 0, 255)
    shifted_img_color_hsv[:,:,2] = np.clip(shifted_img_color_hsv[:,:,2] + value, 0, 255)
    
    # Switch back to uint8 since with clip we went back below 256
    shifted_img_color_rgb = cv2.cvtColor(shifted_img_color_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Show image
    if show:
        plt.imshow(shifted_img_color_rgb, interpolation = 'bicubic')
        plt.axis('off')
        plt.show()
        return
    else:
        return shifted_img_color_rgb

def plot_2_images(first, 
                  second, 
                  mode,
                  first_title = None, 
                  second_title = None, 
                  interpolation = None):

    plt.subplot(1, 2, 1)
    plt.title(first_title)
    if mode=="gray":
        plt.imshow(first, cmap='gray', interpolation = interpolation, vmin=0, vmax=255)
    elif mode=="rgb":
        plt.imshow(first, interpolation = interpolation, vmin=0, vmax=255)
    else: 
        raise ValueError("Unknown mode = {}".format(mode))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(second_title)
    if mode=="gray":
        plt.imshow(second, cmap='gray', interpolation = interpolation, vmin=0, vmax=255)
    elif mode=="rgb":
        plt.imshow(second, interpolation = interpolation, vmin=0, vmax=255)
    else: 
        raise ValueError("Unknown mode = {}".format(mode))
    plt.axis('off')
    plt.show()