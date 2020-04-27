import numpy as np
import cv2
import matplotlib.pyplot as plt

def resize_image(image, factor, interpolation):
    
    assert factor != 0.

    image_h = image.shape[0]
    image_w = image.shape[1]
    
    new_image_h = int(image_h*factor)
    new_image_w = int(image_w*factor)
    
    return cv2.resize(image, (new_image_w, new_image_h), interpolation = interpolation)

def shift_rgb_values(input_color_rgb, red = 0, green = 0, blue = 0):
    
    # Switch to uint16 to be able to go above 256
    shifted_img_color_rgb = input_color_rgb.copy().astype(np.uint16)
    
    # We use clip and not modulo to avoid cycling
    shifted_img_color_rgb[:,:,0] = np.clip(shifted_img_color_rgb[:,:,0] + red, 0, 255)
    shifted_img_color_rgb[:,:,1] = np.clip(shifted_img_color_rgb[:,:,1] + green, 0, 255)
    shifted_img_color_rgb[:,:,2] = np.clip(shifted_img_color_rgb[:,:,2] + blue, 0, 255)
    
    # Switch back to uint8 since with clip we went back below 256
    shifted_img_color_rgb = shifted_img_color_rgb.astype(np.uint8)

    # Show image
    plt.imshow(shifted_img_color_rgb, interpolation = 'bicubic')
    plt.axis('off')
    plt.show()
    
    return

def shift_hsv_values(input_color_hsv, hue = 0, saturation = 0, value = 0):
    
    # Switch to uint16 to be able to go above 256
    shifted_img_color_hsv = input_color_hsv.copy().astype(np.uint16)
    
    # We use clip and not modulo to avoid cycling
    shifted_img_color_hsv[:,:,0] = np.clip(shifted_img_color_hsv[:,:,0] + hue, 0, 179)
    shifted_img_color_hsv[:,:,1] = np.clip(shifted_img_color_hsv[:,:,1] + saturation, 0, 255)
    shifted_img_color_hsv[:,:,2] = np.clip(shifted_img_color_hsv[:,:,2] + value, 0, 255)
    
    # Switch back to uint8 since with clip we went back below 256
    shifted_img_color_rgb = cv2.cvtColor(shifted_img_color_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Show image
    plt.imshow(shifted_img_color_rgb, interpolation = 'bicubic')
    plt.axis('off')
    plt.show()
    
    return

def plot_2_gray_images(first, second, interpolation = None):

    plt.subplot(1, 2, 1)
    plt.imshow(first, cmap='gray', interpolation = interpolation)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(second, cmap='gray', interpolation = interpolation)
    plt.axis('off')
    plt.show()