
from tkinter import Tk, Label, Entry, Button, messagebox
import cv2
import numpy as np
from collections import defaultdict
import threading
import time
from ultralytics import YOLO
import pygame
import sys

pygame.mixer.init()
pygame.mixer.music.load('alerta.mp3')

usuarios = {}

def tocar_alarme():
    pygame.mixer.music.play()

def executar_monitoramento():
    model = YOLO("runs/detect/train6/weights/best.pt")
    video_path = 'video.mp4'
    cap = cv2.VideoCapture(video_path)

    track_history = defaultdict(lambda: [])
    tempo_afogamento = defaultdict(lambda: 0)
    alerta_ativo = defaultdict(lambda: False)
    deixar_rastro = True

    largura_tela = 960
    altura_tela = 600
    mouse_callback_aplicado = False

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
            cv2.putText(img, "PERIGO!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

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

        texto = "SMARTPOOL"
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        tamanho_fonte = 2
        espessura = 5
        cor = (255, 0, 0)
        (tw, th), _ = cv2.getTextSize(texto, fonte, tamanho_fonte, espessura)
        pos_x = (largura_tela - tw) // 2
        pos_y = 70
        cv2.putText(moldura, texto, (pos_x, pos_y), fonte, tamanho_fonte, cor, espessura, lineType=cv2.LINE_AA)

        cv2.rectangle(moldura, (largura_tela - 180, altura_tela - 80), (largura_tela - 40, altura_tela - 40), (0, 0, 255), -1)
        cv2.putText(moldura, "PARAR", (largura_tela - 160, altura_tela - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("SmartPool Monitoramento", moldura)

        if not mouse_callback_aplicado:
            cv2.setMouseCallback("SmartPool Monitoramento", on_mouse)
            mouse_callback_aplicado = True

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def iniciar_login():
    login_window = Tk()
    login_window.title("Login SmartPool")

    Label(login_window, text="Login").grid(row=0)
    Label(login_window, text="Senha").grid(row=1)
    login_entry = Entry(login_window)
    senha_entry = Entry(login_window, show="*")
    login_entry.grid(row=0, column=1)
    senha_entry.grid(row=1, column=1)

    def login():
        login = login_entry.get().strip()
        senha = senha_entry.get().strip()
        if login in usuarios and usuarios[login] == senha:
            messagebox.showinfo("Sucesso", "Login bem-sucedido!")
            login_window.destroy()
            abrir_tela_principal()
        else:
            messagebox.showerror("Erro", "Login ou senha inválidos.")

    Button(login_window, text="Entrar", command=login).grid(row=2, columnspan=2)
    login_window.mainloop()

def iniciar_cadastro():
    cadastro_window = Tk()
    cadastro_window.title("Cadastro SmartPool")

    Label(cadastro_window, text="Login").grid(row=0)
    Label(cadastro_window, text="Senha").grid(row=1)
    login_entry = Entry(cadastro_window)
    senha_entry = Entry(cadastro_window, show="*")
    login_entry.grid(row=0, column=1)
    senha_entry.grid(row=1, column=1)

    def cadastrar():
        login = login_entry.get().strip()
        senha = senha_entry.get().strip()

        if not login or not senha:
            messagebox.showerror("Erro", "Preencha todos os campos.")
            return

        if login in usuarios:
            messagebox.showerror("Erro", "Login já existe.")
            return

        usuarios[login] = senha
        messagebox.showinfo("Sucesso", "Usuário cadastrado com sucesso!")
        cadastro_window.destroy()
        iniciar_login()

    Button(cadastro_window, text="Cadastrar", command=cadastrar).grid(row=2, columnspan=2)
    cadastro_window.mainloop()

def abrir_tela_principal():
    tela = Tk()
    tela.title("SmartPool - Início")
    Label(tela, text="Bem-vindo ao SmartPool").pack(pady=10)
    Button(tela, text="Iniciar Monitoramento", command=lambda: [tela.destroy(), executar_monitoramento()]).pack(pady=5)
    Button(tela, text="Sair", command=lambda: [tela.destroy(), sys.exit()]).pack(pady=5)
    tela.mainloop()

iniciar_cadastro()
