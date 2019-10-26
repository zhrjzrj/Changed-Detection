import os 


path='/home/zrj/deeplabv3plus_zrj/data/HgoData/test_img/'
imgs=os.listdir(path)
with open('/home/zrj/deeplabv3plus_zrj/data/HgoData/ImageSets/val.txt','w') as f :
    for img in imgs:
        img_name=img.split('.')[0]
        f.write(img_name+'\n')