from PIL import Image

#search pixels that aroud specific location pixel is there white or black pixel
#loc: pixel location spacificed for compare
#img: 
#k: kernel size


is_white = lambda pixel: pixel[0]>240 and pixel[1]>240 and pixel[2]>240 
is_black = lambda pixel: pixel[0]<=20 and pixel[1]<=20 and pixel[2]<=20


def pixel_nearby_traverse_PIL(img, loc, k=3):
    kx_s = loc[0] - k//2 if (loc[0] - k//2) > 0 else 0
    kx_e = loc[0] + k//2 if (loc[0] + k//2) < img.shape[0] else img.shape[0]
    ky_s = loc[1] - k//2 if (loc[1] - k//2) > 0 else 0
    ky_e = loc[1] + k//2 if (loc[1] + k//2) < img.shape[1] else img.shape[1]
    for k_x in range(kx_s, kx_e):
        for k_y in range(ky_s, ky_e):
            pixel = img.getpixel((k_x, k_y))
            if(is_white(pixel) or is_black(pixel)):
                return True
    return False

def text_edge_extract_opt_PIL(img_path, img_orig_path, k=3, save_dir=None):
    img_gray = Image.open(img_path)
    img_orig = Image.open(img_orig_path)
    print (img_gray.size)
    print (img_orig.size)

    width = img_gray.size[0]
    height = img_gray.size[1]

    for x in range(0,width):
        for y in range(0,height):
            pixel_gray = img_gray.getpixel((x,y))

            if (pixel_gray == 255):
                if not pixel_nearby_traverse(img_orig, (x,y)):
                img_gray.putpixel((x,y), 0)

def pixel_nearby_traverse_CV(img, loc, k=3):
    kx_s = loc[0] - k//2 if (loc[0] - k//2) > 0 else 0
    kx_e = loc[0] + k//2 if (loc[0] + k//2) < img.shape[0] else img.shape[0]
    ky_s = loc[1] - k//2 if (loc[1] - k//2) > 0 else 0
    ky_e = loc[1] + k//2 if (loc[1] + k//2) < img.shape[1] else img.shape[1]
    for k_x in range(kx_s, kx_e):
        for k_y in range(ky_s, ky_e):
            pixel = img[k_x, k_y]
            if(is_white(pixel) or is_black(pixel)):
                return True
    return False


def text_edge_extract_opt_CV(img_path, img_orig_path, k=3, save_dir=None):
    img_gray = cv.imread(img_path)
    img_orig = Image.imread(img_orig_path)
    print ('Image gray shape:', img_gray.shape)
    print ('Image origin shape:', img_orig.shape)

    width = img_gray.shape[0]
    height = img_gray.shape[1]

    for x in range(0,width):
        for y in range(0,height):

            if (img_gray[x,y] == 255):
                if not pixel_nearby_traverse(img_orig, (x,y), k):
                img_gray[x,y] = 0

    plt.imshow(img_gray, cmap=plt.gray())  
    plt.title('Image text edge extract optimize')
    plt.show()  
    if save_dir:
        cv.imwrite(os.path.join(save_dir, "text_edge_extract_optimize", ".png"), img_gray)

