from PIL import Image
import math
import os

img='/home/zrj/deeplabv3plus_zrj/data/HgoData/test_img/'
label='/home/zrj/deeplabv3plus_zrj/data/HgoData/results/hgo2019/Segmentation/deeplabv3plus_val_cls/'
imgs=os.listdir(img)
for i in imgs:
    name=i.split('.')[0]

    img1 = Image.open(img+i)#图片1
    img2 = Image.open(label+name+'.png')#图片2
    final_img2 = Image.blend(img1, img2, (math.sqrt(5)-1)/2)
    final_img2.save(name+'.jpg')
#该函数的作用是由于 Image.blend()函数只能对像素大小一样的图片进行重叠，故需要对图片进行剪切。
# def cut_img(img, x, y):
#     """
#     函数功能：进行图片裁剪（从中心点出发）
#     :param img: 要裁剪的图片
#     :param x: 需要裁剪的宽度
#     :param y: 需要裁剪的高
#     :return: 返回裁剪后的图片
#     """
#     x_center = img.size[0] / 2
#     y_center = img.size[1] / 2
#     new_x1 = x_center - x//2
#     new_y1 = y_center - y//2
#     new_x2 = x_center + x//2
#     new_y2 = y_center + y//2
#     new_img = img.crop((new_x1, new_y1, new_x2, new_y2))
#     return new_img


#print(img1.size, img2.size)

#取两张图片中最小的图片的像素
# new_x = min(img1.size, img2.size)[0]  
# new_y = min(img1.size, img2.size)[1]

# new_img1 = cut_img(img1, new_x, new_y)
# new_img2 = cut_img(img2, new_x, new_y)
#print(new_img1.size, new_img2.size)

#进行图片重叠  最后一个参数是图片的权值
# final_img2 = Image.blend(new_img1, new_img2, (math.sqrt(5)-1)/2)
# #别问我为什么是  (math.sqrt(5)-1)/2   这个是黄金比例，哈哈！！
# final_img2.save('ttt.jpg')