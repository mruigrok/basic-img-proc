import os
import time
import imgproc as ip

def test_homo(filename):
    rgb = ip.read_in_image(filename)
    yuv = ip.rgb2yuv(rgb)
    for i in range(10):
        yl = i / 10
        yh = 2 + i / 20
        
        homo_img = ip.homomorphic_filter(yuv, 2, yh, 0.5)
        ip.save_image(ip.yuv2rgb(homo_img), filename.split('.')[0] +'_homo_{}.jpg'.format(i))

def time_ahe(filename, N):
    rgb = ip.read_in_image(filename)
    yuv = ip.rgb2yuv(rgb)

    start = time.time()
    ahe_img = ip.ahe(yuv, N)
    ahe_time = time.time() - start

    start =  time.time()
    ahe_img = ip.fast_ahe(yuv, N)
    fahe_time = time.time() - start

    print('AHE time: {}\nFast AHE time: {}'.format(ahe_time, fahe_time))
    ip.save_image(ip.yuv2rgb(ahe_img), filename.split('.')[0] + '_ahe31.jpg')

def run_fast_ahe(filename, window_size):
    rgb = ip.read_in_image(filename)
    #yuv = rgb2yuv(rgb[1000:2000, 1000:2000, :])
    yuv = ip.rgb2yuv(rgb)
    ahe_img = ip.fast_ahe(yuv, window_size)
    ip.save_image(ip.yuv2rgb(ahe_img), filename.split('.')[0] + '_ahe{}.jpg'.format(window_size))

def test_hist(filename):
    rgb = ip.read_in_image(filename)
    yuv = ip.rgb2yuv(rgb)
    hist_img = ip.hist_eq(yuv, 1)
    ip.save_image(ip.yuv2rgb(hist_img), filename.split('.')[0] + '_homo_hist.jpg')

if __name__ == "__main__":
    images = [x for x in os.listdir() if x.endswith('30s.jpg')]
    print(images)
