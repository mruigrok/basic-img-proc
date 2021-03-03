import time
import proj1

def test_homo(filename):
    rgb = proj1.read_in_image(filename)
    yuv = proj1.rgb2yuv(rgb)
    for i in range(10):
        yl = i / 10
        yh = 2 + i / 20
        
        homo_img = proj1.homomorphic_filter(yuv, 2, yh, 0.5)
        proj1.save_image(proj1.yuv2rgb(homo_img), filename.split('.')[0] +'_homo_{}.jpg'.format(i))

def time_ahe(filename):
    rgb = proj1.read_in_image(filename)
    yuv = proj1.rgb2yuv(rgb)

    N = 9
    start = time.time()
    ahe_img = proj1.ahe(yuv, N)
    ahe_time = time.time() - start

    start =  time.time()
    ahe_img = proj1.fast_ahe(yuv, N)
    fahe_time = time.time() - start

    print('AHE time: {}\nFast AHE time: {}'.format(ahe_time, fahe_time))
    proj1.save_image(proj1.yuv2rgb(ahe_img), filename.split('.')[0] + '_ahe31.jpg')

def run_fast_ahe(filename, window_size):
    rgb = proj1.read_in_image(filename)
    #yuv = rgb2yuv(rgb[1000:2000, 1000:2000, :])
    yuv = proj1.rgb2yuv(rgb)
    ahe_img = proj1.fast_ahe(yuv, window_size)
    proj1.save_image(proj1.yuv2rgb(ahe_img), filename.split('.')[0] + '_ahe{}.jpg'.format(window_size))

def test_hist(filename):
    rgb = proj1.read_in_image(filename)
    yuv = proj1.rgb2yuv(rgb)
    hist_img = proj1.hist_eq(yuv, 1)
    proj1.save_image(proj1.yuv2rgb(hist_img), filename.split('.')[0] + '_homo_hist.jpg')

if __name__ == "__main__":
    #run_fast_ahe('monarch.png', 9)
    #run_fast_ahe('monarch.png', 29)
    #run_fast_ahe('monarch.png', 59)

    images = [x for x in os.listdir() if x.endswith('30s.jpg')]
    test_homo(images[3])
    '''
    for image in images:
        rgb = read_in_image(image)
        yuv = rgb2yuv(rgb)
        sig = logit_cor(yuv, .04)
        save_image(yuv2rgb(sig), '{}_sig.jpg'.format(image.split('.')[0]))
'''
    #time_ahe(images[0])
    #time_ahe('download.jfif')
    #for image in images:
    #    test_homo(image)
    '''
    for image in images:
        rgb = read_in_image(image)
        hist_img = hist_eq_3d(rgb)
        save_image(hist_img, image.split('.')[0] + '_hist3d.jpg')
        print('Done {}'.format(image))
        '''

    images2 = [x for x in os.listdir() if x.endswith('homo.jpg')]

    #run_fast_ahe(images[2], 15)
    #test_homo(images[2])
    #test_hist('20147_00_30s_homo_2.jpg')
    #test_hist(images2[0])
    '''
    rgb = read_in_image(images[1])
    yuv = rgb2yuv(rgb)
    #rgb = hist_eq_3d(rgb)
    yuv = hist_eq_3d(yuv)
    rgb = yuv2rgb(yuv)
    save_image(rgb, 'yuv_hist.jpg')
    #save_image(rgb, 'trial1.jpg')
    '''