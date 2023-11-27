import cv2
import mediapipe as mp
import csv

# MediaPipe hands modelini başlatma
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# CSV dosyası için başlık
header = ['label']
for i in range(42):
    header += [f'x{i}', f'y{i}']

# CSV dosyasını aç ve başlıkla yaz
with open('hands_data_nok.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    # Video kaynağını aç (Burada 0, varsayılan kamera)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # MediaPipe işlemi için görüntüyü BGR'dan RGB'ye çevir
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # El algılama
        results = hands.process(image)

        # El verilerini topla
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                row = ['nok']  # Etiketinizi buraya yazın: 'Ok', 'Nok', veya 'Stop'
                for landmark in hand_landmarks.landmark:
                    row.extend(["{:.2f}".format(landmark.x), "{:.2f}".format(landmark.y)])
                
                
                # Eksik veri varsa doldur (her el için 21 nokta)
                while len(row) < 84:  # 1 etiket + 42 nokta * 3 koordinat
                    row.extend([0.00, 0.00])

                writer.writerow(row)

        # Görüntüyü göster
        cv2.imshow('Hands', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
print(len("hands_data_ok.csv"))
# MediaPipe modelini temizle
hands.close()
