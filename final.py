import sys
from PyQt5.QtWidgets import QApplication,QMainWindow, QFileDialog
from main_screen import Ui_MainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2 
import numpy as np 
import os

class MainWindow:
    def __init__(self):
        super().__init__()
        self.link = None  
        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_win)

        #Khai bao nut upload
        self.uic.upload.clicked.connect(self.linkto)
        self.uic.Kapur.clicked.connect(self.processing)
        self.uic.Otsu.clicked.connect(self.processing)
        self.uic.Evaluate.clicked.connect(self.evaluate)
    
    def linkto(self):
        #Tim duong dan
        self.link, _ = QFileDialog.getOpenFileName()
        width = 500
        height = 800
        #Mo anh
        self.uic.label.setPixmap(QPixmap(self.link).scaled(width, height, Qt.KeepAspectRatio)) # set ảnh theo 1 kích thước nhất định
        self.uic.label.setAlignment(Qt.AlignCenter)
    
    def average_filtering(self):
        img = cv2.imread(self.link, 0) 
        if self.uic.Kernel_size.currentText() == "3x3":
            window_size = 3
        elif  self.uic.Kernel_size.currentText() == "5x5":
            window_size = 5
        elif self.uic.Kernel_size.currentText() == "7x7":
            window_size = 7
            
        m, n = img.shape         
         
        mask = np.ones([window_size, window_size], dtype=int)
        mask = mask / 9        
         
        img_new = np.zeros([m, n])        
        for i in range(1, m-1): 
            for j in range(1, n-1): 
                temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2] 
                
                img_new[i, j]= temp                 
        img_new = img_new.astype(np.uint8)
        return img_new
    
    def median_filtering(self):
         
        img_noisy1 = cv2.imread(self.link, 0) 

        if self.uic.Kernel_size.currentText() == "3x3":
            window_size = 3
        elif  self.uic.Kernel_size.currentText() == "5x5":
            window_size = 5
        elif self.uic.Kernel_size.currentText() == "7x7":
            window_size = 7        

        m, n = img_noisy1.shape         

        img_new1 = np.zeros([m, n]) 
        
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                temp = []
                for k in range(window_size):
                    for l in range(window_size):
                        temp.append(img_noisy1[i - 1 + k, j - 1 + l])

                temp = sorted(temp)
                img_new1[i, j] = temp[len(temp) // 2]

        img_new1 = img_new1.astype(np.uint8)

        return img_new1

    def histogram_equalization(self):

        img = cv2.imread(self.link, 0) 
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])

        cdf = hist.cumsum()

        cdf_normalized = (cdf * 255 / cdf[-1]).astype(np.uint8)

        img_equalized = cdf_normalized[img]

        return img_equalized
          
    
    def otsu_segmentation(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist, bins = np.histogram(image.flatten(), 256, [0, 256])

        total_pixels = image.shape[0] * image.shape[1]

        sum_all = 0
        sum_all_sq = 0
        wB = 0
        wF = 0
        max_var = 0
        threshold = 0

        for i in range(256):
            wB += hist[i]
            if wB == 0:
                continue

            wF = total_pixels - wB
            if wF == 0:
                break

            sum_all += i * hist[i]
            sum_all_sq += i**2 * hist[i]

            meanB = sum_all / wB
            meanF = (sum_all_sq - sum_all**2 / wB) / wF

            var_between = wB * wF * (meanB - meanF)**2

            if var_between > max_var:
                max_var = var_between
                threshold = i

        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

        return binary_image
    
    def kapur_segmentation(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist, bins = np.histogram(image.flatten(), 256, [0, 256])

        total_pixels = image.shape[0] * image.shape[1]

        p = hist / total_pixels
        cumulative_sum = np.cumsum(p)
        cumulative_mean = np.cumsum(np.multiply(np.arange(256), p))
        cumulative_mean_sq = np.cumsum(np.multiply(np.arange(256)**2, p))

        max_entropy = -1
        threshold = 0

        for i in range(1, 256):
            w0 = cumulative_sum[i]
            w1 = 1 - w0

            if w0 == 0 or w1 == 0:
                continue

            mean0 = cumulative_mean[i] / w0
            mean1 = (cumulative_mean_sq[i] - cumulative_mean[i]**2 / w0) / w1

            entropy = -w0 * np.log2(w0) - w1 * np.log2(w1)

            if entropy > max_entropy:
                max_entropy = entropy
                threshold = i

        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

        return binary_image
   

    def processing(self):
        if self.uic.Filter.currentText() == "Average_filter":
            img = self.average_filtering()
        elif self.uic.Filter.currentText() == "Median_filterr":
            img = self.median_filtering()
        elif self.uic.Filter.currentText() == "Histogram_equalization":
            img = self.histogram_equalization()
                
        if self.uic.Kapur.click:
            out = self.kapur_segmentation(img)
        elif self.uic.Otsu.click:
            out =self.otsu_segmentation(img)


        file_name_filter = os.path.basename(self.link)
        output_folder_filter = "E:\BMe\XLA_homeless\dataset_individual_homework\Filtered_image/"
        output_file_filter = file_name_filter
        output_path_filter = os.path.join(output_folder_filter, output_file_filter)
        cv2.imwrite(output_path_filter, img)

        file_name_segment = file_name_filter + "_segmented"
        output_folder_segment = "E:\BMe\XLA_homeless\dataset_individual_homework\Segmentation_image/"
        output_file_segment = file_name_segment
        output_path_segment = os.path.join(output_folder_segment, output_file_segment)
        cv2.imwrite(output_path_segment, out)


        width = 500
        height = 800
        #Mo anh
        self.uic.label_7.setPixmap(QPixmap(output_path_filter).scaled(width, height, Qt.KeepAspectRatio)) # set ảnh theo 1 kích thước nhất định
        self.uic.label_7.setAlignment(Qt.AlignCenter)

        self.uic.label_8.setPixmap(QPixmap(output_path_segment).scaled(width, height, Qt.KeepAspectRatio)) # set ảnh theo 1 kích thước nhất định
        self.uic.label_8.setAlignment(Qt.AlignCenter)

        return out

    

    def evaluate(self):
        img = cv2.imread(self.link)
        ground_truth = self.otsu_segmentation(img)
        processed_images = self.processing()
        ground_truth_flat = ground_truth.flatten()
        processed_flat = processed_images.flatten()

        # True Positives, True Negatives, False Positives, False Negatives
        tp = np.sum((ground_truth_flat == 1) & (processed_flat == 1))
        tn = np.sum((ground_truth_flat == 0) & (processed_flat == 0))
        fp = np.sum((ground_truth_flat == 0) & (processed_flat == 1))
        fn = np.sum((ground_truth_flat == 1) & (processed_flat == 0))

        # Sensitivity (True Positive Rate or Recall)
        sensitivity = tp / (tp + fn)

        # Specificity (True Negative Rate)
        specificity =  tn / (tn + fp)

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)

  
        self.uic.lineEdit_3.setText(str(accuracy))
        self.uic.lineEdit_2.setText(str(sensitivity))
        self.uic.lineEdit.setText(str(specificity))

  
    
    def show(self):
        self.main_win.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())