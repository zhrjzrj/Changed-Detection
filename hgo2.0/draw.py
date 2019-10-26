

import cv2
     
he='/home/zrj/Object_detection/hgo3.0/test1.jpg'
he2='/home/zrj/Object_detection/hgo3.0/test2.jpg'
test=cv2.imread(he)
print(test.shape)
test2=cv2.imread(he2)
print(test2.shape)
        # cv2.imshow('test',test)
          # 166.03738 || 257.6172 || 253.83871 || 378.84848


# 1 label: c score: tensor(0.9989) 795.8562 || 574.78534 || 1089.9589 || 707.62115
# 2 label: c score: tensor(0.9797) 264.5744 || 394.16583 || 486.82922 || 541.3974
# 3 label: c score: tensor(0.9653) 786.41895 || 729.3984 || 980.4159 || 918.14307
# 4 label: c score: tensor(0.8939) 373.24026 || 520.6371 || 610.83594 || 738.6943
img2=cv2.rectangle(test,(795,1089),(574,707),(0,0,255),3)
img3=cv2.rectangle(test2,(795,1089),(574,707),(0,0,255),3)
img2=cv2.rectangle(test,(264,486),(394,541),(0,0,255),3)
img3=cv2.rectangle(test2,(264,486),(394,541),(0,0,255),3)
img2=cv2.rectangle(test,(786,980),(729,918),(0,0,255),3)
img3=cv2.rectangle(test2,(786,980),(729,918),(0,0,255),3)
img2=cv2.rectangle(test,(373,610),(520,738),(0,0,255),3)
img3=cv2.rectangle(test2,(373,610),(520,738),(0,0,255),3)
cv2.imwrite("zrj_test_final1.jpg", img2)
cv2.imwrite("zrj_test_final2.jpg", img3)