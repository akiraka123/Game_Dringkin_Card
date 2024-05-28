import cv2
import numpy as np
import copy
import ModulKlasifikasiCitraCNN as mCNN

cap = cv2.VideoCapture(0)
width, height = 200, 300
if not cap.isOpened():
    print("Cannot open camera")
    exit()

LabelKelas = (  "Kosong",    
                "Hati Dua",
                "Hati Tiga",
                "Hati Empat",
                "Hati Lima",
                "Hati Enam",
                "Hati Tujuh",
                "Hati Delapan",
                "Hati Sembilan",
                "Hati Sepuluh",
                "Hati Jack",
                "Hati Queen",
                "Hati King",
                "Hati Ace",
                "Wajik Dua",
                "Wajik Tiga",
                "Wajik Empat",
                "Wajik Lima",
                "Wajik Enam",
                "Wajik Tujuh",
                "Wajik Delapan",
                "Wajik Sembilan",
                "Wajik Sepuluh",
                "Wajik Jack",
                "Wajik Queen",
                "Wajik King",
                "Wajik Ace",
                "Keriting Dua",
                "Keriting Tiga",
                "Keriting Empat",
                "Keriting Lima",
                "Keriting Enam",
                "Keriting Tujuh",
                "Keriting Delapan",
                "Keriting Sembilan",
                "Keriting Sepuluh",
                "Keriting Jack",
                "Keriting Queen",
                "Keriting King",
                "Keriting Ace",
                "Sekop Dua",
                "Sekop Tiga",
                "Sekop Empat",
                "Sekop Lima",
                "Sekop Enam",
                "Sekop Tujuh",
                "Sekop Delapan",
                "Sekop Sembilan",
                "Sekop Sepuluh",
                "Sekop Jack",
                "Sekop Queen",
                "Sekop King",
                "Sekop Ace", 
                )
model = mCNN.LoadModel("BobotKartuKeriting.h5")

def DrawText(img,sText,pos):
    font        = cv2.FONT_HERSHEY_SIMPLEX
    posf        = pos
    fontScale   = .7
    fontColor   = (0,0,255)
    thickness   = 2
    lineType    = 2
    cv2.putText(img,sText,
        posf,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType)
    return copy.deepcopy(img)

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
            
    result = frame.copy()
    cv2.drawContours(result, card_contours, -1, (0, 255, 0), 2)
    
    if len(card_contours) >= 1:
        for card in card_contours:
            # print (card)
            if len(card) == 4:
                if np.linalg.norm(card[:,0][0] - card[:,0][1]) < np.linalg.norm(card[:,0][1] - card[:,0][2]):
                    dst_points = np.array([[width, 0],[0, 0], [0, height], [width, height]], dtype=np.float32)
                else:
                    dst_points = np.array([[0, 0], [0, height],[width, height], [width,0 ] ], dtype=np.float32)
                perspective_matrix = cv2.getPerspectiveTransform(card.astype(np.float32), dst_points)
                warped_card = cv2.warpPerspective(adapthres, perspective_matrix, (width, height))
                warped_card = cv2.cvtColor(warped_card,cv2.COLOR_GRAY2BGR)

                cv2.imshow('warpcard', warped_card)
                # Feed into model
                X = []
                img = cv2.resize(warped_card,(128,128))
                img = np.asarray(img)/255
                img = img.astype('float32')
                X.append(img)
                X = np.array(X)     
                X = X.astype('float32')

                # Predict
                hs = model.predict(X,verbose = 0)
                n = np.max(np.where(hs== hs.max()))

                card_center_x = int((card[1][0][0]))
                card_center_y = int((card[0][0][1] + card[2][0][1]) / 2)

                DrawText(result, f'{LabelKelas[n]} {"{:.2f}".format(hs[0,n])}', [card_center_x, card_center_y])

    cv2.imshow('Original', frame)
    cv2.imshow('threshold', threshold)
    
    # Put text into image

    cv2.imshow('hasil', result)

    if cv2.waitKey(1) & 0xFF == 27:  
        break
    
cap.release()
cv2.destroyAllWindows()
