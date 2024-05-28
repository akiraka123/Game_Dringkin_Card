import cv2
import numpy as np
import copy
import ModulKlasifikasiCitraCNN as mCNN
import time
import random

# Video Capture 
cap = cv2.VideoCapture(0) # 1280 x 720
lebar_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
tinggi_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

key = None
if not cap.isOpened():
    print("Cannot open camera")
    exit()
#model
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
#draw text
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
#fungsi Deteksi kartunya
def deteksiKartu(frame):
    width, height = 200, 300
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.Canny(blurred, 70, 200)
    adapthres = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,71,20)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    card_contours = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(contour) > 5000 and cv2.contourArea(contour) < 90000:
            # print(cv2.contourArea(contour))
            card_contours.append(approx)
            
    cv2.drawContours(frame, card_contours, -1, (255, 0, 0), 2)
    
    if len(card_contours) >= 1:
        for card in card_contours:
            # print (cv2.contourArea(card))
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
                
                DrawText(frame, f'{LabelKelas[n]} {"{:.2f}".format(hs[0,n])}', [card_center_x, card_center_y])
                
                return LabelKelas[n]

# variabel Box untuk game state 
ukuran_kotak = (400, 720)
gambar_kotak = np.zeros((ukuran_kotak[1], ukuran_kotak[0], 3), dtype=np.uint8)
tinggi_kotak = ukuran_kotak[1] // 2  
lebar_kotak = ukuran_kotak [0] // 2
#fungsi untuk memanggil window Game State
def windowStateGame(State,Computer):
    # Kotak pertama
    gambar_kotak[:tinggi_kotak, :] = (255, 255, 255)
    cv2.putText(gambar_kotak, "State", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(gambar_kotak, f'GameTurn : {Nowgameturn}', (100, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(gambar_kotak, State, (40, tinggi_kotak//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Kotak kedua
    gambar_kotak[tinggi_kotak:, :] = (0, 0, 0)  
    cv2.putText(gambar_kotak, "Computer", (10, tinggi_kotak + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(gambar_kotak, f"Computer card: {len(comHandCard)}", (40, (tinggi_kotak*2)- 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(gambar_kotak, Computer, (40, tinggi_kotak + tinggi_kotak//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # Tampilkan gambar kotak
    cv2.imshow('Game State', gambar_kotak)

# Game Master Kontrol
gameTurn = 0
Nowgameturn = 0
gameState = "Draw Card"
cardFound = []

openingCard = None
whosWinTurn = None
isPrevChangeTurn = False

# COM Variable
comDecision = "Waiting"
comHandCard = []
lastDrawcomCard = None
comCard = None

# Player Variable
playerHandCard = 0
isPlayerPlayCardDetected = False
playerCard = None
lastDrawPlayerCard = None

#time variable for Animation
timetake = 20
timecom = 0
timeplayer = 0
timebattle = 0
timeOpen = 0 

#fungsi membuat kotak pada frame untuk menginput kartu ke openingCard
def open_Card(frame):
    global timeOpen,openingCard, gameTurn,gameState,comDecision,Nowgameturn
    hasil_Deteksi= deteksiKartu(frame[50:350,426:826])
    bukaan = cv2.rectangle(frame, (426, 50), (826, 350), (0, 255, 0), timeOpen)
    cv2.putText(bukaan, 'OpenCard', (426, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    if (hasil_Deteksi is not None):
        global playerHandCard
        if (timeOpen >= timetake):
            openingCard = hasil_Deteksi.split()[0]
            gameTurn +=1
            Nowgameturn = gameTurn
            gameState = "Player Turn"
            playerHandCard = len(comHandCard)
            print (len(comHandCard))
        else:
            timeOpen += 1
    else:
        timeOpen = 0

#fungsi untuk komputer melakukan Draw card
def draw_Com(frame):
    global timecom, playerHandCard,lastDrawcomCard
    hasil_Deteksi= deteksiKartu(frame[400:720,0:400])
    drwenemy = cv2.rectangle(frame, (10, 400), (400, 700), (0, 255, 0), timecom)
    cv2.putText(drwenemy, 'Draw Computer Card', (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    if (hasil_Deteksi is not None):
        if (timecom >= timetake):
            if (hasil_Deteksi not in cardFound and lastDrawcomCard == None):
                lastDrawcomCard = hasil_Deteksi
                cardFound.append(hasil_Deteksi)
                print (f"cardfound {cardFound}")
                comHandCard.append(hasil_Deteksi)
                print (comHandCard)    
                
                    
        else:
            timecom += 1
    else:
        lastDrawcomCard = None
        timecom = 0   

#fungsi untuk Player melakukan Draw card
def draw_player(frame):
    global timeplayer,playerHandCard,playerCard,lastDrawPlayerCard
    hasil_Deteksi= deteksiKartu(frame[400:700,853:1270])
    drwplayer = cv2.rectangle(frame, (853, 400), (1270, 700), (0, 255, 0), timeplayer)
    cv2.putText(drwplayer, 'Draw Player Card', (853, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)    
    if (hasil_Deteksi!= None):
        if (timeplayer >= timetake):
            if (lastDrawPlayerCard == None):
                lastDrawPlayerCard = hasil_Deteksi
                playerHandCard += 1
        else:
            timeplayer += 1
    else:
        lastDrawPlayerCard = None
        timeplayer = 0   

# Fungsi Inti dari Permainan dimana player meletakkan kartu yang dipanggil (battle Arena nya)
def battle_arena(frame):
    global timebattle,playerHandCard,isPlayerPlayCardDetected,gameTurn,comCard
    global playerCard, whosWinTurn, Nowgameturn,gameState
    hasil_Deteksi= deteksiKartu(frame[400:700, 426:826]) 
    btlarena = cv2.rectangle(frame, (426, 400), (826, 700), (0, 255, 0), timebattle)
    cv2.putText(btlarena, 'Battle Arena', (426, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (255, 255, 255), 2, cv2.LINE_AA)

    #jika komputer menang dironde sebelumnya
    if (whosWinTurn == "Computer Win Round"):
        if Nowgameturn==gameTurn:
            comCard = comAI(hasil_Deteksi)
            gameTurn+=1

        if (hasil_Deteksi!= None and isPlayerPlayCardDetected == False):
            if (timebattle >= timetake and isPlayerPlayCardDetected == False):
                    isPlayerPlayCardDetected = True
                    playerHandCard-=1
                    playerCard = hasil_Deteksi
                    whosWinTurn = bandingkan_kartu (comCard, playerCard)
                    gameState = whosWinTurn
            else:
                timebattle += 1
        else:
            isPlayerPlayCardDetected = False
            timebattle = 0   
    else: # Jika player menang dironde sebelumnya
        if (Nowgameturn==gameTurn and hasil_Deteksi!= None):
            if (timebattle >= timetake ):
                if (gameTurn != 1 and isPlayerPlayCardDetected == False):
                    isPlayerPlayCardDetected = True
                    gameTurn += 1
                    playerHandCard-=1
                    playerCard = hasil_Deteksi
                    comCard = comAI(playerCard.split()[0])
                    if comCard == 'Draw Card' : #Komputer tidak memiliki kartu dengan bentuk yang sama
                        return
                    whosWinTurn = bandingkan_kartu (comCard, playerCard)
                    gameState = whosWinTurn
                    
                elif gameTurn == 1 and isPlayerPlayCardDetected == False:
                    isPlayerPlayCardDetected = True
                    gameTurn += 1
                    playerHandCard-=1
                    playerCard = hasil_Deteksi
                    comCard = comAI(openingCard)
                    if comCard == 'Draw Card' : #Komputer tidak memiliki kartu dengan bentuk yang sama
                        return
                    whosWinTurn = bandingkan_kartu (comCard, playerCard)
                    gameState = whosWinTurn
                else :  timebattle = 10
            else:
                timebattle += 1 
        else:
            isPlayerPlayCardDetected = False
            timebattle = 0   
# fungsi untuk menentukan siapa yang menang (player atau komputer)
def bandingkan_kartu(com, player):
    kartu = ["Dua", "Tiga", "Empat", "Lima", "Enam", "Tujuh", "Delapan", 
             "Sembilan", "Sepuluh", "Jack", "Queen", "King", "Ace"]
    jenis_com, nilai_com = com.split()
    jenis_player, nilai_player = player.split()
    indeks_com = kartu.index(nilai_com)
    indeks_player = kartu.index(nilai_player)

    if jenis_com == jenis_player:
        if indeks_com < indeks_player:
            print("player win")
            return "Player Win Round"
        elif indeks_com > indeks_player:
            print ("computer win")
            return "Computer Win Round"
#langkah apa yang akan dilakukan komputer dipanggil dengan Fungsi dibawah Ini
def comAI(selectedCard):
    global comDecision
    randomcard = random.randint(0, len(comHandCard) - 1)
    # yang dilakukan jika komputer kalah di ronde sebelumnya
    if selectedCard is not None:
        # Pengecekan apakah Komputer memiliki kartu dengan bentuk yang sama dengan input dari selectedCard
        if any(selectedCard in kartu for kartu in comHandCard):
            kartu_bukaan = [kartu for kartu in comHandCard if selectedCard in kartu]
            kartu_tertinggi = max(kartu_bukaan, key=lambda kartu: kartu_bukaan.index(kartu))
            comHandCard.remove(kartu_tertinggi)
            
            print(f"Komputer memilih {kartu_tertinggi} sebagai kartu bukaan tertinggi.")
            comDecision = kartu_tertinggi
            return kartu_tertinggi
        else:
            # print(f"Komputer tidak memiliki kartu {selectedCard}.")
            comDecision = "Draw Card"
            return 'Draw Card'
    else: # yang dilakukan jika komputer menang di ronde sebelumnya
        comSelect = comHandCard[randomcard]
        print(f"Komputer memilih {comSelect} sebagai kartu bukaan tertinggi.")
        comHandCard.remove(comSelect)
        comDecision = comSelect
        return comSelect
#Main game
while True:
    ret, frame = cap.read()
    if not ret:
        break
    #Pembagian kartu untuk komputer
    elif Nowgameturn == 0:  
        windowStateGame(gameState ,comDecision)
        draw_Com(frame)
        open_Card(frame)
    #jika Player menang    
    elif playerHandCard == 0:
        A= 'Player Win'
        windowStateGame(A,A)
        cv2.putText(frame, 'PLAYER WIN', (lebar_frame//6-50, tinggi_frame-100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)

    #jika Komputer menang 
    elif len(comHandCard) == 0:
        A= 'Computer Win'
        windowStateGame(A,A)

        cv2.putText(frame, 'COMPUTER WIN', (lebar_frame//6-50, tinggi_frame-100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)

    #jika Komputer tidak memiliki kartu yang sama dengan bentuknya
    elif comCard == 'Draw Card' : 
        draw_Com(frame)
        comCard = comAI (playerCard.split()[0])
        windowStateGame('Draw Card',"waiting")
        if comCard != 'Draw Card' :
             
            whosWinTurn = bandingkan_kartu (comCard, playerCard)
            gameState = whosWinTurn
    #START GAMEEEEEE !!!  
    elif Nowgameturn !=0:
        windowStateGame(gameState ,comDecision)
        battle_arena(frame)
        draw_player(frame)

    
    if comDecision in LabelKelas: 
        kartuimageCom = cv2.imread (f"GmbrKartu/{comDecision}.png")
        kartuimageCom = cv2.resize(kartuimageCom,(200,300))
        x1 = lebar_frame -250
        y1 = 50
        cv2.putText(frame, 'kartu Komputer', (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (255, 255, 255), 2, cv2.LINE_AA)
        frame[y1:y1+300, x1:x1+200] = kartuimageCom
        
        if comDecision.split()[0] == playerCard.split()[0]:
            kartuimageplayer = cv2.imread (f"GmbrKartu/{playerCard}.png")
            kartuimageplayer = cv2.resize(kartuimageplayer,(200,300))
            x2 = x1 -250
            y2 = 50
            frame[y2:y2+300, x2:x2+200] = kartuimageplayer
            cv2.putText(frame, 'kartu player', (x2-10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255, 255, 255), 2, cv2.LINE_AA)
        
    cv2.imshow('Permainan Kartu Minum', frame)
    
    key = cv2.waitKey(1)
    if key == 27: #Exit ( press ESC Keyboard )
        break
    if key == 32: #Go Next Turn ( ronde selanjutnya ) ( press SPACE Keyboard )
        Nowgameturn = gameTurn 
        print (f'gameTurn {Nowgameturn}')
        isPlayerPlayCardDetected = False
        comDecision = "waiting"
        gameState = 'Turn Player'
        playerCard = 'Kosong'
    
cap.release()

cv2.destroyAllWindows()
