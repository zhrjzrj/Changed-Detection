# Hgo2.0

## Find the location of changed area

this code is based on SSD, and make some modification

the  trained module are two:one in pre-trained weight about vgg16, another is trained for this task named hgo.pth

https://pan.baidu.com/s/10FhqwDrUKA585ZVc_mWK9A （k05j ）

eval is used to save the result of every input, the confidence and the coordinate of the boxes

the format of input is similarly to voc, so the dataset code is based on VOCdataset

发现自己英语真的菜，第一次写，可能连我自己都看不明白，还是用中文写吧。

给出两个权重，一个是vgg16的预训练权重，另一个是在hgo图片上训练好的权重，可以直接拿来检测，整个代码是基于SSD改的，改了一些参数，网络结构略作调整，整体在我的task上还是work的



### Test

可以用给的几个example来测试一下：

结果见result

