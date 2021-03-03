import cv2
import math
import numpy as np
import os
import sys

# CE 4TN4 Project 1
# Matthew Ruigrok

def read_in_image(filename):
    if os.path.exists(filename):
        img = cv2.imread(filename)
        img_rgb = bgr2rgb(img)
        return img_rgb
    else:
        print('Couldn''t find file!')
        return None

def save_image(img, outfilename):
    img_bgr = rgb2bgr(img)
    cv2.imwrite(outfilename, img_bgr)
    return

def rgb2bgr(rgb):
    m = np.array([[0., 0., 1.],
                 [0., 1., 0.],
                 [1., 0., 0.]
                 ])
    bgr = np.dot(rgb, m)
    return bgr

# Just swap r and b values; rgb2bgr and rgb2bgr are the same transformation function
bgr2rgb = rgb2bgr

def rgb2yuv(rgb):
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]
                 ])
    yuv = np.dot(rgb,m)
    yuv[:,:,1:] += 128.0
    return yuv

def yuv2rgb(yuv):
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235]
                 ])
    rgb = np.dot(yuv,m)
    rgb[:,:,0] -= 179.45477266423404
    rgb[:,:,1] += 135.45870971679688
    rgb[:,:,2] -= 226.8183044444304
    rgb = rgb.clip(0,255).astype(np.uint8)
    return rgb

# Gray level transformations - gamma correction, log correction, piecewise linear, and inverse sigmoid
# O(MN)
def gam_cor(img, c, gamma):
    gam = np.copy(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            gam[i][j][0] = c * (img[i][j][0] ** gamma)
    return gam

# O(MN)
def log_cor(img, c):
    log = np.copy(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            log[i][j][0] = c * math.log(1 + img[i][j][0], 10)
    return log

# O(MN)
def pw_linear(img, alpha, beta, gamma, r1, r2):
    if r1 > r2:
        print('Invalid r1 and r2 values')
        return img
    pw = np.copy(img)
    s1 = alpha * r1
    s2 = beta * (r2 - r1) + s1
    for i in range(len(img)):
        for j in range(len(img[0])):
            r = img[i][j][0]
            if r <= r1:
                pw[i][j][0] = alpha * r
            elif r <= r2:
                pw[i][j][0] = beta * (r - r1) + s1
            else:
                pw[i][j][0] = gamma * (r - r2) + s2
    return pw

# O(MN)
def logit_cor(img, k, gray_levels=256):
    sig = np.clip(np.copy(img), 1, gray_levels-1)
    for i in range(len(img)):
        for j in range(len(img[0])):
            x = gray_levels/sig[i][j][0] - 1
            if x <= 0:
                sig[i][j][0] = 0
            else:
                sig[i][j][0] = -k * math.log(x) + 127
    return sig

# Global histogram equilization - either 1D (just Y vals) or 3D
# O(MN)
def hist_eq(img, ndim, gray_levels=256):
    if ndim != 1 and ndim != 3:
        print('ndim should be 1 or 3')
        return img

    # Make sure values are in range, grab image size
    rows, cols = img.shape[0], img.shape[1]
    hist_img = np.copy(img)
    hist_img = hist_img.clip(0, gray_levels-1).astype(np.uint8)

    if ndim == 1:
        # 1D histogram equalization
        # Get the 1d image histogram, find the mapping function and
        # then update the pixel values
        pixel_map = np.zeros(gray_levels)
        pixel_sum = 0
        sum_const = (gray_levels - 1)/ (rows * cols) # Constant value for CDF
        for i in range(rows):
            for j in range(cols):
                pixel_map[hist_img[i][j][0]] += 1
        
        for i, count in enumerate(pixel_map):
            pixel_sum += count
            pixel_map[i] = pixel_sum * sum_const

        for i in range(rows):
            for j in range(cols):
                hist_img[i][j][0] = pixel_map[hist_img[i][j][0]]
           
    elif ndim == 3:
        # 3D histogram equalization
        # Get the 3d image histogram, find the mapping function and
        # then update the pixel values
        r_map = np.zeros(gray_levels)
        g_map = np.zeros(gray_levels)
        b_map = np.zeros(gray_levels) 
        r_sum, g_sum, b_sum = 0, 0, 0
        sum_const = (gray_levels - 1)/ (rows * cols) # constant value for CDF
        for i in range(rows):
            for j in range(cols):
                r_map[hist_img[i][j][0]] += 1
                g_map[hist_img[i][j][1]] += 1
                b_map[hist_img[i][j][2]] += 1
        
        for i in range(len(r_map)):
            r_sum += r_map[i]
            g_sum += g_map[i]
            b_sum += b_map[i]
            r_map[i] = r_sum * sum_const
            g_map[i] = g_sum * sum_const
            b_map[i] = b_sum * sum_const

        for i in range(rows):
            for j in range(cols):
                hist_img[i][j][0] = r_map[hist_img[i][j][0]]
                hist_img[i][j][1] = g_map[hist_img[i][j][1]]
                hist_img[i][j][2] = b_map[hist_img[i][j][2]]

    return hist_img

# Adaptive histogram equilization, window_size should be and odd number
# O(MNW^2)
def ahe(img, window_size, gray_levels=256):
    if window_size % 2 == 0:
        print('window_size must be an odd number')
        return img

    # Copy the image
    rows, cols, w = img.shape[0], img.shape[1], window_size
    ahe_img = np.clip(np.copy(img), 0, 255).astype(np.uint8)
    img_copy = np.copy(ahe_img)

    # Pad the copy image with for AHE - 'reflect' gives nice edges
    img_copy = np.pad(img_copy, ((w//2,w//2), (w//2,w//2), (0,0)), 'reflect') 
    sum_const = (gray_levels - 1) / (w ** 2)
    print('AHE Progress')
    for i in range(rows):
        for j in range(cols):
            hist = np.zeros(gray_levels)
            for m in range(-w//2, w//2):
                for n in range(-w//2, w//2):
                    hist[img_copy[i+m+w//2][j+n+w//2][0]] += 1           
            pixel_sum = hist[:img_copy[i+w//2][j+w//2][0] + 1].sum()
            ahe_img[i][j][0] = pixel_sum * sum_const

        # Progress bar for sanity check
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*((20*i)//(rows-1)), 5*((20*i)//(rows-1))))
        sys.stdout.flush()

    return ahe_img

# Sped up histogram equilization by not recomputing the entire histogram
# O(MW^2 + MNW)
def fast_ahe(img, window_size, gray_levels=256):
    if window_size % 2 == 0:
        print('window_size must be an odd number')
        return img

    # Copy the image
    rows, cols, w = img.shape[0], img.shape[1], window_size
    ahe_img = np.clip(np.copy(img), 0, 255).astype(np.uint8)
    img_copy = np.copy(ahe_img)

    # Pad the copy image with for AHE - 'reflect' gives nice edges
    img_copy = np.pad(img_copy, ((w//2,w//2), (w//2,w//2), (0,0)), 'reflect') 
    sum_const = (gray_levels - 1) / (w ** 2)
    for i in range(rows):
        hist = np.zeros(gray_levels)
        for j in range(cols):
            # New histogram for the new row
            if j == 0:
                for m in range(-w//2, w//2):
                    for n in range(-w//2, w//2):
                        hist[img_copy[i+m+w//2][j+n+w//2][0]] += 1
            # Just delete the left column and insert the right column to the existing histogram
            else:
                for m in range(-w//2, w//2):
                    hist[img_copy[i+m+w//2][j][0]] -= 1
                    hist[img_copy[i+m+w//2][j+w-1][0]] += 1
            
            #Find the sum of pixels up to a certain value
            #pixel_sum = hist[:img_copy[i+w//2][j+w//2][0] + 1].sum()
            pixel_sum = hist[:img_copy[i+w//2][j+w//2][0]].sum()
            ahe_img[i][j][0] = pixel_sum * sum_const

        # Progress bar for sanity check
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*((20*i)//(rows-1)), 5*((20*i)//(rows-1))))
        sys.stdout.flush()

    return ahe_img

# f_cu is the cutoff frequency, gam_h and gam_l are parameters
# O(MN*log(MN))
def homomorphic_filter(img, f_cu, gam_h, gam_l):
    # Grab only the grayscale portion of the image
    homo_img = np.copy(img[:,:,0])
    min_nonzero = np.min(homo_img[np.nonzero(homo_img)])
    homo_img[homo_img <= 0] = min_nonzero # Avoid zero values

    # Take log of image, then FFT, and then shift to the centre
    homo_img_log = np.log(np.float64(homo_img), dtype=np.float64)
    dft = np.fft.fft2(homo_img_log, axes=(0,1))
    dft_shift = np.fft.fftshift(dft)

    # Create a high pass filter
    radius = f_cu
    mask = np.zeros_like(homo_img, dtype=np.float64)
    cx, cy = mask.shape[1] // 2, mask.shape[0] // 2
    cv2.circle(mask, (cx,cy), radius, 1, -1)

    # Anti-aliasing via blurring and multiply the mask
    mask = cv2.GaussianBlur(mask, (157,157), 0)
    mask = (gam_h - gam_l) * (1 - mask) - gam_l
    dft_shift_filtered = np.multiply(dft_shift, mask)
    dft_shift_filtered = np.fft.ifftshift(dft_shift_filtered)

    # Convert back to spatial domain
    homo_img_back = np.fft.ifft2(dft_shift_filtered, axes=(0,1))
    homo_img_back = np.abs(homo_img_back)
    homo_img_back = np.exp(homo_img_back, dtype=np.float64)
    homo_img_back = np.clip(homo_img_back, 0, 255)

    # Now add the homo filtered image (Y) back to the UV components and return complete YUV image
    homo_img = np.dstack((homo_img_back, img[:,:,1], img[:,:,2]))
    return homo_img

# Run all the graylevel transformation functions
def run_all_graylevel(filename):
    # Get RGB values of input image and convert to YUV
    rgb = read_in_image(filename)
    yuv = rgb2yuv(rgb)
    outfile = filename.split('.jpg')[0]

    # Perform hist, log, gamma transformations
    log_img = log_cor(yuv, 80)
    save_image(yuv2rgb(log_img), outfile + '_lt.jpg')
    print('Done log transform')
    gam_img = gam_cor(yuv, 8, 0.6)
    save_image(yuv2rgb(gam_img), outfile + '_gc.jpg')
    print('Done gamma correction')
    pw_img = pw_linear(yuv, 5, 0.6, 0.25, 20, 200)
    save_image(yuv2rgb(pw_img), outfile + '_pwl.jpg')
    print('Done piecewise linear')
    logit_img = logit_cor(yuv, 25)
    save_image(yuv2rgb(logit_img), outfile + '_logit.jpg')
    print('Done logit')

# To print for incorrect command line parameters
def print_options():
    print('')
    print('Please specify what you want to do and run again')
    print('------------- Options ---------------')
    print('-gl  [image] ==> run gray-level transforms on an image')
    print('-ghe [image] ==> run global histogram equalization on an image')
    print('-lhe [image] [window] ==> run local histogram equalization with window size \'window\'')
    print('-hf  [image] [gamma h] [gamma l] ==> run homomorphic filter with \'gamma h\' and \'gamma l\'')
    print('Note : Not specfying an [image] will run on all .jpg images in directory')
    print('')

# Main program - follow the guide given to use the functions
# Runs from the command line, specify the processing you want to do and the parameters
if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) == 1:
        print_options()
        exit(0)

    if sys.argv[1] == '-gl':   # Gray level transforms
        if len(sys.argv) == 2: # Run all 30s.jpg test images in current directory
            images = [x for x in os.listdir() if x.endswith('30s.jpg')]
            for image in images:
                run_all_graylevel(image)
                print('Done {}'.format(image))
        else:
            run_all_graylevel(sys.argv[2])
            print('Done {}'.format(sys.argv[2]))

    elif sys.argv[1] == '-ghe': # Global histogram equalization
        if len(sys.argv) == 2: # Run all .jpg test images in current directory
            images = [x for x in os.listdir() if x.endswith('30s.jpg')]
            for image in images:
                yuv = rgb2yuv(read_in_image(image))
                yuv = hist_eq(yuv, 1)
                save_image(yuv2rgb(yuv), image.split('.')[0] + '_he.jpg')
                print('Done {}'.format(image))
        else:
            yuv = rgb2yuv(read_in_image(sys.argv[2]))
            yuv = hist_eq(yuv, 1)
            save_image(yuv2rgb(yuv), sys.argv[2].split('.')[0] + '_ghe.jpg')
            print('Done {}'.format(sys.argv[2]))

    elif sys.argv[1] == '-lhe': # Local histogram equalization
        if len(sys.argv) == 2: # No window size given
            print('No window size given, please specify')
        elif len(sys.argv) == 3 and not sys.argv[2].endswith('.jpg'): # Run all .jpg test images in current directory
            window = int(sys.argv[2])
            images = [x for x in os.listdir() if x.endswith('30s.jpg')]
            for image in images:
                yuv = rgb2yuv(read_in_image(image))
                yuv = fast_ahe(yuv, window)
                save_image(yuv2rgb(yuv), image.split('.')[0] + '_lhe_{}.jpg'.format(window))
                print('Done {}'.format(image))
        elif len(sys.argv) == 3 and sys.argv[2].endswith('.jpg'):
            print('Specify a window size')
        elif len(sys.argv) == 4:
            window = int(sys.argv[3])
            yuv = rgb2yuv(read_in_image(sys.argv[2]))
            yuv = fast_ahe(yuv, window)
            save_image(yuv2rgb(yuv), sys.argv[2].split('.')[0] + '_lhe_{}.jpg'.format(window))
            print('Done {}'.format(sys.argv[2]))

    elif sys.argv[1] == '-hf': # Homomorphic filter
        if len(sys.argv) <= 3 or (len(sys.argv) == 4 and sys.argv[2].endswith('.jpg')):
            print('gamma h and/or gamma l not specified, running with default 2.0 and 0.5')
            if sys.argv[2].endswith('.jpg'):
                yuv = rgb2yuv(read_in_image(sys.argv[2]))
                yuv = homomorphic_filter(yuv, 1, 2.0, 0.5)
                save_image(yuv2rgb(yuv), sys.argv[2].split('.')[0] + '_hf_{}_{}.jpg'.format(2.0, 0.5))
                print('Done {}'.format(sys.argv[2]))
            else:
                print('need to pass an image')
        elif len(sys.argv) == 4:
            gam_h, gam_l = float(sys.argv[2]), float(sys.argv[3])
            images = [x for x in os.listdir() if x.endswith('30s.jpg')]
            for image in images:
                yuv = rgb2yuv(read_in_image(image))
                yuv = homomorphic_filter(yuv, 1, gam_h, gam_l)
                save_image(yuv2rgb(yuv), image.split('.')[0] + '_hf_{}_{}.jpg'.format(gam_h, gam_l))
                print('Done {}'.format(image))
        elif len(sys.argv) == 5:
            gam_h, gam_l = float(sys.argv[3]), float(sys.argv[4])
            yuv = rgb2yuv(read_in_image(sys.argv[2]))
            yuv = homomorphic_filter(yuv, 1, gam_h, gam_l)
            save_image(yuv2rgb(yuv), sys.argv[2].split('.')[0] + '_hf_{}_{}.jpg'.format(gam_h, gam_l))
            print('Done {}'.format(sys.argv[2]))
    else:
        print_options()
        print('Command not found....exiting')
        exit(0)
