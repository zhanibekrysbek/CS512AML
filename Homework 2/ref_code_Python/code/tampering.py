import cv2
import numpy as np

def translate_img(img_vec, shift_tuple):
    img = np.reshape(img_vec, newshape=(16, 8))
    rows,cols = img.shape

    M = np.float32([[1,0,shift_tuple[0]],[0,1,shift_tuple[1]]])
    dst = cv2.warpAffine(img, M, (cols,rows))
    dst = np.reshape(dst, newshape=(rows*cols,))
    return dst

def rotate_img(img_vec, angle):
    img = np.reshape(img_vec, newshape=(16, 8))
    rows, cols = img.shape

    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(img, M, (cols, rows))

    dst = np.reshape(dst, newshape=(rows*cols,))
    return dst


if __name__ == '__main__':
    train_file = 'train.txt'
    train_pre = []
    train_img = []
    istream = open(train_file,'r')
    for eachline in istream:
        allparts = eachline.strip().split(' ')
        train_pre.append(allparts[:5])
        train_img.append([int(i) for i in allparts[5:]])
    istream.close()

    transform_file = 'transform.txt'
    transform_list = []
    istream = open(transform_file, 'r')
    for eachline in istream:
        allparts = eachline.strip().split(' ')
        transform_list.append(allparts)
    istream.close()

    first_x_lines = [500, 1000, 1500, 2000]

    for eachx in first_x_lines:
        transform_first_list = transform_list[:eachx]
        transform_dict = {}
        for eachtrans in transform_first_list:
            if len(eachtrans) == 3:
                transform_dict[int(eachtrans[1])] = [eachtrans[0], int(eachtrans[2])]
            else:
                transform_dict[int(eachtrans[1])] = [eachtrans[0], (int(eachtrans[2]), int(eachtrans[3]))]

        for ind, pre in enumerate(train_pre):
            word_id = int(pre[3])
            if word_id in transform_dict:
                trans_info = transform_dict[word_id]
                img_vec = np.array(train_img[ind], dtype=np.uint16)

                if trans_info[0] == 'r':
                    print('rotate')
                    angle = trans_info[1]
                    dst = rotate_img(img_vec, angle)
                    dst = dst.tolist()
                    train_img[ind] = dst
                else:
                    print('translate')
                    shift_tuple = trans_info[1]
                    print(shift_tuple)
                    dst = translate_img(img_vec, shift_tuple)
                    dst = dst.tolist()
                    train_img[ind] = dst
        
        ostream = open('tampering_%d.txt' % (eachx), 'w')
        for ind, newvec in enumerate(train_img):
            pre = train_pre[ind]
            for eachpre in pre:
                ostream.write(eachpre + ' ')
            for ind in range(len(newvec)-1):
                ostream.write(str(newvec[ind]) + ' ')
            ostream.write(str(newvec[len(newvec)-1]) + '\n')
        ostream.close()
