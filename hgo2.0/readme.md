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

给的三组图片是效果比较好的，

![1572069899130](C:\Users\64641\AppData\Roaming\Typora\typora-user-images\1572069899130.png)

![1572069909316](C:\Users\64641\AppData\Roaming\Typora\typora-user-images\1572069909316.png)



因为训练数据比较少，也没有另外加其他的trick，所以大部分效果是可以检测出来，但是box的尺寸不太正确，以及位置不是很准确

![1572069924375](C:\Users\64641\AppData\Roaming\Typora\typora-user-images\1572069924375.png)

![1572069928628](C:\Users\64641\AppData\Roaming\Typora\typora-user-images\1572069928628.png)

![1572069951108](C:\Users\64641\AppData\Roaming\Typora\typora-user-images\1572069951108.png)

当然还有一些效果不好的

![1572069964256](C:\Users\64641\AppData\Roaming\Typora\typora-user-images\1572069964256.png)

![1572069968229](C:\Users\64641\AppData\Roaming\Typora\typora-user-images\1572069968229.png)