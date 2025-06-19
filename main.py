from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np
import time
import threading
import pygame

# Som
pygame.mixer.init()
pygame.mixer.music.load('alerta.mp3')

def tocar_alarme():
    pygame.mixer.music.play()

# Vídeo
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# Carregando o modelo treinado do YOLO
model = YOLO("runs/detect/train6/weights/best.pt")

# Rastreamento e alerta
track_history = defaultdict(lambda: [])
tempo_afogamento = defaultdict(lambda: 0)
alerta_ativo = defaultdict(lambda: False)

# Configurações da tela
largura_tela = 960
altura_tela = 600

deixar_rastro = True
mouse_callback_aplicado = False  # Evita erro ao setar callback antes da janela existir

# Botão parar
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if largura_tela - 180 <= x <= largura_tela - 40 and altura_tela - 80 <= y <= altura_tela - 40:
            print("Botão PARAR pressionado")
            cap.release()
            cv2.destroyAllWindows()
            exit()

while True:
    ret, img = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model.track(img, persist=True)
    alerta_visivel = False

    for result in results:
        img = result.plot()

        if deixar_rastro:
            try:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()
                nomes_classes = model.names

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    nome_classe = nomes_classes[class_id]
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)

                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)

                    if nome_classe.lower() == "afogando":
                        if tempo_afogamento[track_id] == 0:
                            tempo_afogamento[track_id] = time.time()
                        elif time.time() - tempo_afogamento[track_id] > 5 and not alerta_ativo[track_id]:
                            alerta_ativo[track_id] = True
                            threading.Thread(target=tocar_alarme, daemon=True).start()
                        if alerta_ativo[track_id]:
                            alerta_visivel = True
                    else:
                        tempo_afogamento[track_id] = 0
                        alerta_ativo[track_id] = False

            except Exception as e:
                print("Erro:", e)

    if alerta_visivel and int(time.time() * 2) % 2 == 0: 
        cv2.putText(img, "PERIGO!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Moldura e tamanho do vídeo
    moldura = np.full((altura_tela, largura_tela, 3), 255, dtype=np.uint8)

    altura_video, largura_video = img.shape[:2]
    altura_max_video = altura_tela - 150
    largura_max_video = largura_tela - 100

    escala = min(largura_max_video / largura_video, altura_max_video / altura_video)
    novo_tamanho = (int(largura_video * escala), int(altura_video * escala))
    img_redimensionado = cv2.resize(img, novo_tamanho)

    x_offset = (largura_tela - novo_tamanho[0]) // 2
    y_offset = 100
    moldura[y_offset:y_offset+novo_tamanho[1], x_offset:x_offset+novo_tamanho[0]] = img_redimensionado

    # Título 
    texto = "SMARTPOOL"
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    tamanho_fonte = 2
    espessura = 5
    cor = (255, 0, 0)
    (tw, th), _ = cv2.getTextSize(texto, fonte, tamanho_fonte, espessura)
    pos_x = (largura_tela - tw) // 2
    pos_y = 70
    cv2.putText(moldura, texto, (pos_x, pos_y), fonte, tamanho_fonte, cor, espessura, lineType=cv2.LINE_AA)

    # Botão para parar
    cv2.rectangle(moldura, (largura_tela - 180, altura_tela - 80), (largura_tela - 40, altura_tela - 40), (0, 0, 255), -1)
    cv2.putText(moldura, "PARAR", (largura_tela - 160, altura_tela - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Abrindo a janela
    cv2.imshow("SmartPool Monitoramento", moldura)

    # Para clicar no botão
    if not mouse_callback_aplicado:
        cv2.setMouseCallback("SmartPool Monitoramento", on_mouse)
        mouse_callback_aplicado = True

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
