import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np 
from torchvision import models


#device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

#Hyper parameter 설정 
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

def load_emoji():
    smile_emoji = cv2.imread('project/emoji_prj/intel/smile.png', cv2.IMREAD_UNCHANGED)
    smile_emoji = cv2.resize(smile_emoji, (100, 100))

    # 이모티콘 투명도 처리
    alpha_emoji = smile_emoji[:, :, 3] / 255.0
    color_emoji = smile_emoji[:, :, :3]

    # 이모티콘 이미지 읽기
    neutral_emoji = cv2.imread('project/emoji_prj/intel/neutral.png', cv2.IMREAD_UNCHANGED)
    neutral_emoji = cv2.resize(neutral_emoji, (100, 100))

    # 이모티콘 투명도 처리
    alpha_emoji2 = neutral_emoji[:, :, 3] / 255.0
    color_emoji2 = neutral_emoji[:, :, :3]
    return alpha_emoji,color_emoji,alpha_emoji2,color_emoji2

def model_load():
    # 모델 불러오기
    model = models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=7)
    model.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # pre trained model
    model.load_state_dict(torch.load("project/emoji_prj/intel/best_model.pt", 
                                     map_location=torch.device('cpu')))### need to change
    model.to(device)
    return model

def model_pred(model,img,frame):

    pred = torch.stack([torch.tensor(im, dtype=torch.float32) for im in img])  # (N, 48, 48)
    pred = pred.permute(0, 3, 1, 2)
    pred_loader = DataLoader(pred, batch_size = BATCH_SIZE, shuffle=True) #drop_last = True

    for epoch in range(EPOCHS):
        with torch.no_grad():
            for img in pred_loader:
                pred = model(img.to(device))
                _, predicted = torch.max(pred.data, 1)  # 예측된 라벨

                h, w, _ = frame.shape
                roi = frame[h-100:h, w-100:w]
             # clear_output()
            if predicted==3:
                # print("T'm Happy!")
                # 이모티콘을 프레임에 합성
                np.multiply(1.0 - alpha_emoji[:, :, np.newaxis], roi, out=roi, casting="unsafe")
                np.add(color_emoji * alpha_emoji[:, :, np.newaxis], roi, out=roi, casting="unsafe")
            else:
                # 이모티콘을 프레임에 합성
                np.multiply(1.0 - alpha_emoji2[:, :, np.newaxis], roi, out=roi, casting="unsafe")
                np.add(color_emoji2 * alpha_emoji2[:, :, np.newaxis], roi, out=roi, casting="unsafe")
            # else:
            #     continue

def webcam():
    #웹캠에서 영상을 읽어온다
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) #WIDTH
    cap.set(4, 480) #HEIGHT
 
    #얼굴 인식 캐스케이드 파일 읽는다
    face_cascade = cv2.CascadeClassifier('project/emoji_prj/intel/haarcascade_frontalface_alt.xml')
    count = 0
    model_input_size = (48, 48) # 모델에 입력되는 이미지 크기
    while(True):
    # frame 별로 capture 한다
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #인식된 얼굴 갯수를 출력
        print(len(faces))

    # 인식된 얼굴에 사각형을 출력한다
        for (x,y,w,h) in faces:
             cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

          # 추출된 얼굴 영역 이미지를 모델에 입력할 준비
             face_img = gray[y:y+h, x:x+w]
             face_img = cv2.resize(face_img, model_input_size)
             face_img = np.expand_dims(face_img, axis=0)
         # 모델에 입력되는 이미지는 일반적으로 채널 수가 3이기 때문에 채널 수를 1로 바꿔줌
             face_img = np.expand_dims(face_img, axis=-1) 

         # 모델에 입력하여 결과 출력
             model_pred(pre_model,face_img,frame)
         

    #화면에 출력한다
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

alpha_emoji, color_emoji, alpha_emoji2, color_emoji2 = load_emoji()
pre_model = model_load()
webcam()