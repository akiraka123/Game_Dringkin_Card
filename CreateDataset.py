import cv2
import os
import datetime
import time
import ModulKlasifikasiCitraCNN as mCNN
import numpy as np

width, height = 200, 300
def GetFileName():
    x = datetime.datetime.now()
    s = x.strftime('%Y-%m-%d-%H%M%S%f')
    return s

def CreateDir(path):
    ls = []
    head_tail = os.path.split(path)
    ls.append(path)
    while len(head_tail[1])>0:
        head_tail = os.path.split(path)
        path = head_tail[0]
        ls.append(path)
        head_tail = os.path.split(path)
    for i in range(len(ls)-2,-1,-1):
        sf = ls[i]
        isExist = os.path.exists(sf)
        if not isExist:
            os.makedirs(sf)


def CreateDataSet(sDirektoriData,sKelas,NoKamera,FrameRate):
    sDirektoriKelas = sDirektoriData+"/"+sKelas
    CreateDir(sDirektoriKelas)

    # For webcam input:
    cap = cv2.VideoCapture(NoKamera)
    TimeStart = time.time()

    # For limiting 30 second
    time30start = time.time()

    # For start taking pics
    isSaving = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        threshold = cv2.Canny(blurred, 70, 200)
        adapthres = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,71,20)

        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        card_contours = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4 and cv2.contourArea(contour) > 5000:
                card_contours.append(approx)
                
        isDetected = False
        result = frame.copy()
        cv2.drawContours(result, card_contours, -1, (0, 255, 0), 2)

        if len(card_contours) == 1:
            card = card_contours[0]
            if len(card) == 4:
                if np.linalg.norm(card[:,0][0] - card[:,0][1]) < np.linalg.norm(card[:,0][1] - card[:,0][2]):
                    dst_points = np.array([[width, 0],[0, 0], [0, height], [width, height]], dtype=np.float32)
                else:
                    dst_points = np.array([[0, 0], [0, height],[width, height], [width,0 ] ], dtype=np.float32)
                perspective_matrix = cv2.getPerspectiveTransform(card.astype(np.float32), dst_points)
                warped_card = cv2.warpPerspective(adapthres, perspective_matrix, (width, height))
                
                cv2.imshow('warpcard', warped_card)
                isDetected = True
                

        cv2.imshow('Original', frame)
        cv2.imshow('threshold', threshold)
        cv2.imshow('hasil', result)
        
        TimeNow = time.time()
        if TimeNow-TimeStart>1/FrameRate:
            sfFile = sDirektoriKelas+"/"+GetFileName()
            if isSaving and isDetected:
                cv2.imwrite(sfFile+'.jpg', warped_card)
            TimeStart = TimeNow
        
        text = "Record"
        position = (0, 30)  # (0, 0) coordinates at the left top corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 255)  # Color in BGR format (green in this case)
        font_thickness = 2

        if isSaving:
            cv2.putText(threshold, text, position, font, font_scale, font_color, font_thickness)
        

        key = cv2.waitKey(5)

        if key == 32:
            isSaving = not isSaving

        if time.time() - time30start >= 30:
            break

        if key & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

DirektoriDataSet = "Kartu"

# CreateDataSet(DirektoriDataSet, "Wajik Ace", NoKamera=0, FrameRate=20)
# CreateDataSet(DirektoriDataSet, "Wajik Dua", NoKamera=0, FrameRate=20)
# CreateDataSet(DirektoriDataSet, "Wajik Tiga", NoKamera=0, FrameRate=20)
# CreateDataSet(DirektoriDataSet, "Wajik Empat", NoKamera=0, FrameRate=20)
# CreateDataSet(DirektoriDataSet, "Wajik Lima", NoKamera=0, FrameRate=20)
# CreateDataSet(DirektoriDataSet, "Wajik Enam", NoKamera=0, FrameRate=20)
# CreateDataSet(DirektoriDataSet, "Wajik Tujuh", NoKamera=0, FrameRate=20)
# CreateDataSet(DirektoriDataSet, "Wajik Delapan", NoKamera=0, FrameRate=20)
# CreateDataSet(DirektoriDataSet, "Wajik Sembilan", NoKamera=0, FrameRate=20)
# CreateDataSet(DirektoriDataSet, "Wajik Sepuluh", NoKamera=0, FrameRate=20)
# CreateDataSet(DirektoriDataSet, "Wajik Jack", NoKamera=0, FrameRate=20)
# CreateDataSet(DirektoriDataSet, "Wajik Queen", NoKamera=0, FrameRate=20)
CreateDataSet(DirektoriDataSet, "Wajik King", NoKamera=0, FrameRate=20)
