import numpy as np
from scipy.ndimage import convolve
from skimage.io import imsave

def to_grayscale(img):
    return 0.299*img[..., 0] + 0.587*img[..., 1] + 0.114*img[..., 2]


def convolve_naive(x):
    res = np.zeros(shape=x.shape, dtype=np.float64)

    for i in range(0, x.shape[0]):
        res[i, 0] = x[i, 1] - x[i, 0]
        res[i, x.shape[1]-1] = x[i, x.shape[1]-1] - x[i, x.shape[1]-2]

    for i in range(0, x.shape[0]):
        for j in range(1, x.shape[1]-1):
            res[i, j] = x[i, j+1] - x[i, j-1]

    return res


def seam_carve(image, mode, mask):
    args = mode.split()

    if "vertical" in args:
        image = image.swapaxes(0,1)
        if not mask is None:
            mask = mask.swapaxes(0,1)
    
    gray = to_grayscale(image)

    # grad_y = convolve_naive(gray)
    # grad_x = convolve_naive(gray.transpose())
    # grad = np.sqrt(grad_y**2 + grad_x.transpose()**2)

    grad = np.sqrt(convolve(gray, [[-1, 0, 1]], mode='nearest')**2 + \
                 convolve(gray, [[-1], [0], [1]], mode='nearest')**2)

    #print(np.abs(grad-grad_naive))

    

    max_grad = grad.shape[0]*grad.shape[1]*256
    grad[mask==1] += max_grad
    grad[mask==-1] += -max_grad

    # print(grad[mask==-1])
    # print(grad)
    
    # dbg = grad-grad.min()
    # dbg = dbg/dbg.max()
    # imsave("dbg.jpg", dbg)



    mat = np.zeros(shape=grad.shape, dtype=np.float64)
    mat[0, :] = grad[0, :]
    for i in range(1, mat.shape[0]):
        j = 0
        mat[i, j] = grad[i,j] + min(mat[i-1, j], mat[i-1, j+1])
        j = mat.shape[1]-1
        mat[i, j] = grad[i,j] + min(mat[i-1, j], mat[i-1, j-1])
        for j in range(1, mat.shape[1]-1):
            mat[i,j] = grad[i,j] + min(mat[i-1, j-1], mat[i-1, j], mat[i-1, j+1])

    indices = np.zeros(shape=(mat.shape[0],), dtype=np.int32)
    
    sort_idx = np.argsort(mat[-1])
    indices[-1] = np.argmin(mat[-1])

    for i in range(mat.shape[0]-2, -1, -1):

        pb = indices[i+1]
        if pb == 0:
            indices[i] = np.argmin([mat[i, pb], mat[i, pb+1]])
        elif pb == mat.shape[1]-1:
            indices[i] = pb - np.argmin([mat[i, pb], mat[i, pb-1]])
        else:
            indices[i] = pb - 1 + np.argmin([mat[i, pb-1], mat[i, pb], mat[i, pb+1]])


    bool_mask = np.ones(shape=mat.shape, dtype=np.bool)
    indices_rows = np.arange(indices.shape[0])
    bool_mask[indices_rows, indices] = False


    if "shrink" in mode.split():
        res = image[bool_mask].reshape(image.shape[0], -1, 3)
        if not mask is None:
            res_mask = mask[bool_mask].reshape(mask.shape[0], -1)
        else:
            res_mask = None
    else:
        res = np.zeros(shape=(image.shape[0], image.shape[1]+1, 3), dtype=np.uint8)
        
        bool_mask_tmp = np.ones(shape=(image.shape[0], image.shape[1]+1), dtype=np.bool)
        bool_mask_tmp[indices_rows, indices] = False
        bool_mask_tmp[indices_rows, indices+1] = False
        
        #res[bool_mask_tmp] = image[bool_mask]
        for ch_i in range(3):
            res[:,:,ch_i][np.where(bool_mask_tmp)] = image[:,:,ch_i][np.where(bool_mask)]

        res[indices_rows, indices] = image[indices_rows, indices]
        res[indices_rows, indices+1] = image[indices_rows, indices]
        res = res.reshape(image.shape[0], -1, 3)

        if not mask is None:
            res_mask = np.zeros(shape=(mask.shape[0], mask.shape[1]+1), dtype=np.uint8)

            res_mask[np.where(bool_mask_tmp)] = mask[np.where(bool_mask)]
            res_mask[indices_rows, indices] = mask[indices_rows, indices]
            res_mask[indices_rows, indices+1] = mask[indices_rows, indices]
            res_mask = res.reshape(mask.shape[0], -1)
        else:
            res_mask = None

    
    

    if "vertical" in args:
        res = res.swapaxes(0,1)
        bool_mask = bool_mask.swapaxes(0,1)
        if not mask is None:
            res_mask = res_mask.swapaxes(0,1)

    imsave("res.jpg", res)

    res_idx = tuple(zip(indices_rows, indices))

    return res, res_mask, (~bool_mask).astype(np.int32)





from sys import argv
def run_single_test(data_dir):
    from numpy import where
    from os.path import join
    from pickle import dump
    from seam_carve import seam_carve
    from skimage.io import imread

    def get_seam_coords(seam_mask):
        coords = where(seam_mask)
        t = [i for i in zip(coords[0], coords[1])]
        t.sort(key=lambda i: i[0])
        return tuple(t)

    def convert_img_to_mask(img):
        return ((img[:, :, 0] != 0) * -1 + (img[:, :, 1] != 0)).astype('int8')

    img = imread(join(data_dir, 'img.png'))
    mask = convert_img_to_mask(imread(join(data_dir, 'mask.png')))

    m = mask
    im = img

    #for m in (None, mask,):
    for i in range(250):
        im, m, _ = seam_carve(im, 'shrink horizontal', mask=m)

                # pred = get_seam_coords(seam)

                # for kek in pred:
                #     print(kek)

if __name__=="__main__":
    run_single_test(argv[1])