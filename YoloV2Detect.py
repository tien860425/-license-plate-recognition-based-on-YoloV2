from keras.preprocessing import image
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import colorsys
import argparse
import os

class Detector(object):
    def __init__(self,  h5_file, classes, box_per_cell=5):
        self.classes = classes
        self.num_classes = len(self.classes)
        self.image_size = 416
        self.cell_size = 13
        self.threshold = 0.5
        # self.anchor = [5.3693, 5.5328, 5.6435, 5.7891, 5.8703,11.6895,11.4611,11.5134,11.5194,11.6127]
        # self.anchor  = [1.08,  3.42,  6.63,  9.42,  16.62,1.19, 4.41,  11.38,  5.11,  10.52]  #前5個為x，後五個為y
        # self.anchor = [5.3693,11.6895]
        self.box_per_cell = box_per_cell
        self.model = load_model(h5_file)
        self.colors = self.random_colors(len(self.classes))
        if self.box_per_cell==1:
            self.anchor = [5.3693, 11.6895]
        else:
            self.anchor = [1.08, 3.42, 6.63, 9.42, 16.62, 1.19, 4.41, 11.38, 5.11, 10.52]
            # self.anchor =  [5.3693, 5.5328, 5.6435, 5.7891, 5.8703,11.6895, 11.4611, 11.5134,11.5194,11.6127]  # 前5個為x，後五個為y
        print('Restore model from:  '+ h5_file)

    def detect(self, image):
        image_h, image_w, _ = image.shape
        # image = cv2.resize(image, (self.image_size, self.image_size))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image = image / 255.0
        # image = np.reshape(image, [1, self.image_size, self.image_size, 3])

        # resize image
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size,self.image_size))
        #
        # # normalize歸一化
        image = image.astype(np.float32) / 225.0
        #
        # # 增加一個維度在第0維——batch_size
        image = np.expand_dims(image, axis=0)

        output = self.model.predict(image)

        results = self.calc_output(output)

        for i in range(len(results)):
            results[i][1] *= (1.0 * image_w / self.image_size)
            results[i][2] *= (1.0 * image_h / self.image_size)
            results[i][3] *= (1.0 * image_w / self.image_size)
            results[i][4] *= (1.0 * image_h / self.image_size)

        return results


    def calc_output(self, output):
        output = np.reshape(output, [self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes])
        boxes = np.reshape(output[:, :, :, :4], [self.cell_size, self.cell_size, self.box_per_cell, 4])    #boxes coordinate
        boxes = self.get_boxes(boxes) * self.image_size

        confidence = np.reshape(output[:, :, :, 4], [self.cell_size, self.cell_size, self.box_per_cell])    #the confidence of the each anchor boxes
        confidence = 1.0 / (1.0 + np.exp(-1.0 * confidence))
        confidence = np.tile(np.expand_dims(confidence, 3), (1, 1, 1, self.num_classes))

        classes = np.reshape(output[:, :, :, 5:], [self.cell_size, self.cell_size, self.box_per_cell, self.num_classes])    #classes
        classes = np.exp(classes) / np.tile(np.expand_dims(np.sum(np.exp(classes), axis=3), axis=3), (1, 1, 1, self.num_classes))

        probs = classes * confidence

        filter_probs = np.array(probs >= self.threshold, dtype = 'bool')
        filter_index = np.nonzero(filter_probs)
        box_filter = boxes[filter_index[0], filter_index[1], filter_index[2]]
        probs_filter = probs[filter_probs]
        classes_num = np.argmax(filter_probs, axis = 3)[filter_index[0], filter_index[1], filter_index[2]]

        sort_num = np.array(np.argsort(probs_filter))[::-1]
        box_filter = box_filter[sort_num]
        probs_filter = probs_filter[sort_num]
        classes_num = classes_num[sort_num]

        for i in range(len(probs_filter)):
            if probs_filter[i] == 0:
                continue
            for j in range(i+1, len(probs_filter)):
                if self.calc_iou(box_filter[i], box_filter[j]) > 0.5:
                    probs_filter[j] = 0.0

        filter_probs = np.array(probs_filter > 0, dtype = 'bool')
        probs_filter = probs_filter[filter_probs]
        box_filter = box_filter[filter_probs]
        classes_num = classes_num[filter_probs]

        results = []
        for i in range(len(probs_filter)):
            results.append([self.classes[classes_num[i]], box_filter[i][0], box_filter[i][1],
                            box_filter[i][2], box_filter[i][3], probs_filter[i],classes_num[i]])

        return results

    def get_boxes(self, boxes):  #input boxes (13,13,5,4)
        #(65, 13)->(5, 13, 13)->(13, 13, 5)
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.box_per_cell),
                                         [self.box_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))
        boxes1 = np.stack([(1.0 / (1.0 + np.exp(-1.0 * boxes[:, :, :, 0])) + offset) / self.cell_size,
                           (1.0 / (1.0 + np.exp(-1.0 * boxes[:, :, :, 1])) + np.transpose(offset, (1, 0, 2))) / self.cell_size,
                           np.exp(boxes[:, :, :, 2]) * np.reshape(self.anchor[:self.box_per_cell], [1, 1, self.box_per_cell]) / self.cell_size,
                           np.exp(boxes[:, :, :, 3]) * np.reshape(self.anchor[self.box_per_cell:], [1, 1, self.box_per_cell]) / self.cell_size])

        return np.transpose(boxes1, (1, 2, 3, 0))


    def calc_iou(self, box1, box2):
        width = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        height = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])

        if width <= 0 or height <= 0:
            intersection = 0
        else:
            intersection = width * height

        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)
        
    def random_colors(self, N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]

        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)) #第一個參數 function 以參數序列中的每一個元素調用 function 函數
        np.random.shuffle(colors)
        return colors


    def draw(self, image, result):
        image_h, image_w, _ = image.shape
        # colors = self.random_colors(len(result))
        for i in range(len(result)):
            xmin = max(int(result[i][1] - 0.5 * result[i][3]), 0)
            ymin = max(int(result[i][2] - 0.5 * result[i][4]), 0)
            xmax = min(int(result[i][1] + 0.5 * result[i][3]), image_w)
            ymax = min(int(result[i][2] + 0.5 * result[i][4]), image_h)
            color = tuple([rgb * 255 for rgb in self.colors[result[i][6]]])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,255), 2)   #cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度).
            #各參數依次是：照片/添加的文字/左上角座標/字體/字體大小/顏色/字體粗細
            cv2.putText(image, result[i][0] + ':%.2f' % result[i][5], (xmin + 1, ymin -5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.6*image_w/256, (0,255,255), 2)


    def image_detect(self, image):
        result = self.detect(image)
        for i in range(len(result)):
            print( '%s:%.2f%%' % (result[i][0],result[i][5] * 100))
            return result

    def video_detect(self, cap):
        while(1):
            ret, image = cap.read()
            if not ret:
                print('Cannot capture images from device')
                break

            result = self.detect(image)
            self.draw(image, result)
            cv2.imshow('Image', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


def main(show=True):
    
    from timeit import default_timer as timer
    start = timer()  # Start Time
    classes=['Plate']
    detector = Detector('yolo-car2.h5',classes, 5)
    classesC=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
    detectorC = Detector('yolo-DirectRec2.h5',classesC, 5)
    # print(detector.model.summary())
    # from keras.utils.vis_utils import plot_model
    # plot_model(detector.model, to_file='yoloStructure.png', show_shapes=True,show_layer_names=True)
    count=0
    for file in os.listdir('./testImage/'):
        imagename = os.path.join("./testImage", file)
        image = cv2.imread(imagename)
        image_h, image_w, _ = image.shape
        result=detector.image_detect(image)
        if result ==None:
            continue
        for i in range(len(result)):
            xmin = max(int(result[i][1] - 0.5 * result[i][3]), 0)
            ymin = max(int(result[i][2] - 0.5 * result[i][4]), 0)
            xmax = min(int(result[i][1] + 0.5 * result[i][3]), image_w)
            ymax = min(int(result[i][2] + 0.5 * result[i][4]), image_h)
            cropImg = image[ymin:ymax,xmin:xmax]
            stdPlate=cv2.resize(cropImg,(190,80),interpolation=cv2.INTER_CUBIC)
            rslt=detectorC.image_detect(stdPlate)
            # sorted(rslt, key=itemgetter(1))
            rslt=sorted(rslt, key=lambda inL: inL[1])
            carNo=''
            for j in range(len(rslt)):
                carNo+=rslt[j][0]
            result[i][0]=carNo
        if len(result) > 0:
            count+=1
            if show:
                detector.draw(image, result)
                cv2.namedWindow(imagename, 0)
                cv2.resizeWindow(imagename, 1024, 768);
                cv2.imshow(imagename, image)
                if cv2.waitKey(0) & 0xFF == ord('q'):   
                    cv2.destroyAllWindows()
                    break
                cv2.destroyAllWindows()
    end = timer()
    print(end - start)  # Total Time
    print('Total samples: %d\nTotal Time %10.3f\nAverage time consume per pictiure: %5.3f' % (count,(end - start),(end - start)/count))
    #detect the image
    # imagename = 'a.jpg'
    # detector.image_detect(imagename)

if __name__ == '__main__':
    main()
