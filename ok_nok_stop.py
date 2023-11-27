import cv2
import joblib
import mediapipe as mp
import numpy as np

# Modeli yükle
model = joblib.load('svm_model.joblib')

# MediaPipe hands modelini başlat
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Video kaynağını başlat
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Görüntüyü çevir ve el işleme için hazırla
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # El algılama
    results = hands.process(image)

    # Görüntüyü RGB'den BGR'ye çevir ve yazdırma için hazırla
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Algılanan her el için işaretleme yap
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # El işaretleri verisini hazırla
            lm_list = [np.array([lm.x, lm.y]) for lm in hand_landmarks.landmark]
            lm_array = np.array(lm_list).flatten()

            # Eğer tek el algılanmışsa ve model iki elin verisiyle eğitilmişse, sıfırlarla doldur
            if len(lm_array) == 42: # 21 nokta * 2 koordinat = 42 özellik
                lm_array = np.concatenate([lm_array, np.zeros(42)])  # İkinci elin verileri için 42 sıfır ekle

            # Modeli kullanarak el hareketini tahmin et
            pred = model.predict([lm_array])
            gesture_label = pred[0]

            print("pred::", pred)

            # Tahmin edilen hareketi ekranda göster
            cv2.putText(image, gesture_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # El işaretlerini çiz
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Görüntüyü göster
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC tuşu ile çık
        break

# Her şeyi serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
