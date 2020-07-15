import cv2
import os
import numpy as np
def resize_image(image_path,image_size,image_format):
    image = cv2.imread(image_path, 1)
    if image_format == 'bgr':
        # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #用这个的话不能imwrite
        image=image[:,:,::-1]
    # image = cv2.imread(image_path,0)
    # height, width = image.shape
    height, width, _ = image.shape
    if height > width:
        scale = image_size / height
        resized_height = image_size
        resized_width = int(width * scale)
    else:
        scale = image_size / width
        resized_height = int(height * scale)
        resized_width = image_size
    image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    #三通道图
    new_image = np.zeros((image_size, image_size, 3))
    new_image[0:resized_height, 0:resized_width] = image #注意这里没有居中直接把新图放到了左上角
    #单通道图
    # new_image = np.zeros((image_size, image_size))
    # new_image[0:resized_height, 0:resized_width] = image
    # new_image = new_image.reshape((*(new_image.shape), 1))
    return new_image

def main():
    image_paths=['infrared.jpg']
    image_size = 416
    image_format='bgr'
    for i in range(len(image_paths)):
        image_path=image_paths[i]
        new_image = resize_image(image_path, image_size, image_format)
        if image_format=='rgb':
            cv2.imwrite(f'resized_{i}.jpg',new_image)
            print('{}成功转换为resized_{}.jpg'.format(image_path.split('//')[-1], i))
        elif image_format == 'bgr':
            cv2.imwrite(f'bgr_resized_{i}.jpg',new_image)
            print('{}成功转换为bgr_resized_{}.jpg'.format(image_path.split('//')[-1], i))

#直接图片的张量用cv2.convert或者是[:,:,::-1]转换bgr 用在nnie项目有问题 还是用博主用过的方法
#https://blog.csdn.net/qq_34533248/article/details/102497297
def nnie_bgr(imgpath,saveimg,save_img_size): 
    img = cv2.imread(imgpath)
    if img is None:
        print("img is none")
    else:
        img = cv2.resize(img,(save_img_size,save_img_size))
        (B, G, R) = cv2.split(img)
        with open(saveimg,'wb')as fp:
            for i in range(save_img_size):
                for j in range(save_img_size):
                    fp.write(B[i, j])
            for i in range(save_img_size):
                for j in range(save_img_size):
                    fp.write(G[i, j])
            for i in range(save_img_size):
                for j in range(save_img_size):
                    fp.write(R[i, j])
        print("save success")

imgpath = "infrared.jpg"
saveimg = "infrared_416x416.bgr"
save_img_size = 416
nnie_bgr(imgpath,saveimg,save_img_size)