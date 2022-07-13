# 車牌辨識
運用YOLO V2訓練的車牌偵測器先找出車牌。再將擷取的車牌影像，使用YOLO V2訓練的字元偵測器直辨識出車牌影像中的字元。本專案的模型檔利用官網指引訓練。再轉為.h5。

## 車牌位置辨識模型訓練
本專案依照官網 ( https://github.com/AlexeyAB/darknet)說明，以MSVS 2015, CUDA 10.0, cuDNN 7.4 及 OpenCV 3.42將darknet.sln及其原始碼建置成darknet.exe。此執行檔即可用來訓練和預測自己想辨識的物件。訓練步驟：
1. 修改yolo-voc.2.0.cfg組態檔裡filters=30， (classes+cords+1)*num_anchors, classes=1。另存於cfg\yolo_car2.cfg
2.	產生一文字檔，obj.names，將欲訓練辨識物件名稱寫入，一列寫一個名稱。在此我們的目標僅辨識車牌，因此這個檔案只有一列寫入plate。存放在data\obj.names。
3.	產生一文字檔，car2.data，裡面包含：
	classes= 1
	train  = data/cartrain.txt
	valid  = data/cartest.txt
	names = data/obj.names
	backup = backup/
4.	將所有參與學習的影像檔(*.jpg)存放於data/img/。
5.	針對每一存放於data/img/的jpg檔，產生相同檔名的*.txt檔，並存放在相同目錄。裡面存放物件編號（第一類編號0），該物件座標。該影像裡有預辨識的物件，每物件一列，<object-class> <x> <y> <width> <height>。
	<object-class>  - 從0到（classes-1）的整數
	<x> <y> <width> <height>  - 浮點值，相對於圖像的寬度和高度，它可以等於0.0到1.0
	例如：<x> = <absolute_x> / <image_width>或<height> = <absolute_height> / <image_height>
	注意：<x> <y>  - 是矩形的中心（不是左上角）

![訓練範例圖做標籤](D:\personel\1.png) 
**訓練範例圖做標籤**
	 官網https://github.com/AlexeyAB/Yolo_mark，提供製作學習範例做此標籤的工具，Yolo_mark.exe。

6.	在data\目錄下，產生一文字檔cartrain.txt，將訓練的每ㄧ影像檔路徑寫入，一個影像檔一列。
	Data\img\img1.jpg
	Data\img\img2.jpg
	Data\img\img3.jpg
7.	下載預訓練好的卷積層權重，並存放到defaultWeight\目錄。http://pjreddie.com/media/files/darknet19_448.conv.23。
8.	開始訓練：
	在darknet.exe目錄下開啟console。
	鍵入下列指令：
darknet.exe detector train data\car.data cfg\yolo-car2.cfg  defaultWeight\darknet19_448.conv.23 backup\

![2](D:\personel\2.png)

在6898 iterations時截圖

![3](D:\personel\3.png)

iterations=20000，avg. loss=0.0023
我們可看出訓練一開始loss由4.5以上到截圖時已降致0.0036，共訓練了64x6898=441472。其中64為batch，一次訓練的批量。
每100次會儲存當時權重，yolo_car2_100.weights，yolo_car2_200.weights，yolo_car2_300.weights，直到45000次，會儲存成yolo_car2_final.weights。

## 車牌位置辨識模型訓練
將34個字元就分成34類（I,O排除），在標註標籤時類別代號就不是只有0，會依照0、1、2…9、A、B、…、Z順序編號。在yolo組態檔中均與偵測車牌、偵測車牌字元位置的組態檔大致相同，除了配合偵測34類，所以classes=34，及輸出的tensor，所以filter=(5+34)*5=195。

![4](D:\personel\4.png) 

yolo_mark標註車牌34類字元位置

![5](D:\personel\5.png)

標註標籤檔案，包括物件編號及4個幾何位置，編號視框的字元而不同。

![6](D:\personel\6.png)

charDirectRec.data檔內容

![7](D:\personel\7.png)

traincharDirectRec.txt內容

traincharDirectRec.txt內容為參與訓練的圖檔路徑，標籤檔如圖60，也在同目錄。
charDirectRec.names檔內容就是要偵測物件的名稱，第一列為第一類物件名稱，第二列為第二類物件名稱，…。
用下列指令開始訓練：
darknet.exe detector train data\charDirectRec.data cfg\yolo-DirectRec2.cfg  defaultWeight\darknet19_448.conv.23

## 將原cfg及weights檔轉成keras的h5.
用[YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K) 將兩個模型轉成yolo-car2.h5及yolo-DirectRec2.h5[下載](https://drive.google.com/drive/folders/11QChpWhNpD4SC8tcTBLat3MbUI2FO8IR?usp=sharing)。

