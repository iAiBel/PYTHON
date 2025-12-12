import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import json, os, threading, traceback
import winsound 
import time 


CAMINHO_BASE = os.path.dirname(os.path.abspath(__file__)) 

# Configs de Caminho
CAMINHO_IMG = os.path.join(CAMINHO_BASE, "imagens")
FOGUETE_PARADO_CAMINHO = os.path.join(CAMINHO_IMG, "personagem", "rocket_parado.png") 
FOGUETE_FOGO_CAMINHO = os.path.join(CAMINHO_IMG, "personagem", "rocket.png") 
CAMINHO_PLANETAS = [os.path.join(CAMINHO_IMG, f"planet{i}.png") for i in range(1, 13)] # 12 planetas
CAMINHO_MUSICA = os.path.join(CAMINHO_BASE, "musicas") 

JAZZ_CAMINHO = os.path.join(CAMINHO_MUSICA, "Jazz.wav")
FOGUETE_SOM_CAMINHO = os.path.join(CAMINHO_MUSICA, "Rocket.wav")

FUNDO_CAMINHO = os.path.join(CAMINHO_IMG, "parque.png")
ARQUIVO_PERGUNTAS = "perguntas.json"

# Arquivo do placar
ARQUIVO_RANKING = "ranking.json" 
print("CAMINHO ONDE O CÓDIGO PROCURA O ARQUIVO:")
print(os.path.join(CAMINHO_BASE, ARQUIVO_PERGUNTAS))

# Pega as perguntas do JSON
try:
    # CORREÇÃO: Usa o caminho base para encontrar o arquivo (Linha corrigida para [Errno 2])
    caminho_completo_perguntas = os.path.join(CAMINHO_BASE, ARQUIVO_PERGUNTAS) 
    with open(caminho_completo_perguntas, 'r', encoding='utf-8') as f:
        perguntas = json.load(f)
        if not isinstance(perguntas, list):
            perguntas = [] 
except Exception as erro:
    print("Erro ao carregar perguntas:", erro)
    perguntas = []

# Variáveis 
jogo_ligado = False 
em_pergunta = False 
posicao_alvo = 1 #a nave e perguntas iniciam no planeta 1 (índice 0 é o ponto de partida)
vidas = 3
pontos = 0
direcao_movimento = [0, 0] 
tempo_inicio = 0 

# Controle do som
tem_jazz = False 
som_foguete = False 

# IDs do Canvas 
tela_jogo_canvas = None
id_foguete = None
ids_planetas = []
id_destaque = None 
id_hud_vidas = None
id_hud_pontos = None 

# Funções de Hover dos Botões 

# Botões Secundários (volta para o cinza)
def on_enter_secundario(event):
    """Muda a cor de fundo para verde vibrante."""
    event.widget.config(bg="#94EA80", fg="#000000") 

def on_leave_secundario(event):
    """Volta a cor de fundo para cinza escuro."""
    event.widget.config(bg="#3C3F41", fg="#FFFFFF") 

# Botões Principais (volta para o verde/amarelo original)
def on_enter_principal(event):
    """Muda a cor de fundo para um verde/amarelo ainda mais claro e brilhante."""
    event.widget.config(bg="#E0FF66", fg="#000000") 

def on_leave_principal(event):
    """Volta para o verde/amarelo original do ESTILO_BOTAO_PRINCIPAL."""
    event.widget.config(bg="#C6E043", fg="#000000") 


# Som
def toca_jazz_se_nao_tiver_rolando():
    global tem_jazz
    if tem_jazz: return
    if not os.path.exists(JAZZ_CAMINHO): return
    try:
        winsound.PlaySound(JAZZ_CAMINHO, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP) 
        tem_jazz = True
    except Exception as erro:
        print(f"Erro ao tocar Jazz: {erro}")
        pass

def para_todos_os_sons():
    global tem_jazz, som_foguete
    winsound.PlaySound(None, winsound.SND_PURGE) 
    tem_jazz = False
    som_foguete = False

# Recorde do mesmo jogador
def salva_e_ordena_ranking(nome_jogador, pontos, tempo_segundos):
    ranking = []
    nome_jogador = nome_jogador.strip()

    if os.path.exists(ARQUIVO_RANKING):
        try:
            with open(ARQUIVO_RANKING, 'r', encoding='utf-8') as f:
                ranking = json.load(f)
        except Exception as e:
            print(f"Erro ao ler ranking: {e}. Lista vazia.")
            ranking = []

    novo_resultado = {
        "nome": nome_jogador,
        "pontos": pontos,
        "tempo": int(tempo_segundos) 
    }
    
    encontrado = False
    for i, registro in enumerate(ranking):
        if registro['nome'].lower() == nome_jogador.lower():
            encontrado = True
            
            if pontos > registro['pontos']:
                ranking[i] = novo_resultado
                break
            elif pontos == registro['pontos'] and tempo_segundos < registro['tempo']:
                ranking[i] = novo_resultado
                break
            else:
                break
    
    if not encontrado:
        ranking.append(novo_resultado)

    ranking.sort(key=lambda x: (x['pontos'], -x['tempo']), reverse=True)

    try:
        with open(ARQUIVO_RANKING, 'w', encoding='utf-8') as f:
            json.dump(ranking, f, indent=4)
        print("Ranking salvo e ordenado com atualização de score.")
    except Exception as e:
        print(f"Erro ao salvar ranking: {e}")

# rANKING: ler e mostrar placar
def mostra_ranking():
    ranking = []
    
    if os.path.exists(ARQUIVO_RANKING):
        try:
            with open(ARQUIVO_RANKING, 'r', encoding='utf-8') as f:
                ranking = json.load(f)
        except:
            pass 
            
    modal_ranking = tk.Toplevel(janela_principal)
    modal_ranking.title("Ranking de Comandantes Estelares") 
    modal_ranking.transient(janela_principal)
    modal_ranking.grab_set()
    modal_ranking.focus_force()

    tk.Label(modal_ranking, text="🏆 Top Comandantes Estelares 🏆", font=("Arial", 18, "bold")).pack(pady=10)
    
    ranking_texto = ""
    for i, registro in enumerate(ranking[:10], 1): 
        minutos = registro['tempo'] // 60
        segundos = registro['tempo'] % 60
        
        ranking_texto += f"#{i: <2} {registro['pontos']:>3} pts | {minutos:02d}:{segundos:02d}s | {registro['nome']}\n"
    
    if not ranking_texto:
        ranking_texto = "Ainda não temos resultados."
        
    tk.Label(modal_ranking, text=ranking_texto, justify=tk.LEFT, font=("Courier", 14)).pack(padx=20, pady=5)
    
    tk.Button(modal_ranking, text="Fechar", command=modal_ranking.destroy).pack(pady=10)

# Configuração da janela
janela_principal = tk.Tk()
janela_principal.title("Via Láctea da Estatística")
janela_principal.attributes("-fullscreen", True)

def sair_tela_cheia(evento=None):
    janela_principal.attributes("-fullscreen", False)

janela_principal.bind("<Escape>", sair_tela_cheia)
janela_principal.update_idletasks()
LARGURA_TELA = janela_principal.winfo_screenwidth()
ALTURA_TELA = janela_principal.winfo_screenheight()

# Posições dos planetas
POSICOES_BASE = [
    (150, 500),  (450, 200),  (350, 850),  (700, 500), 
    (900, 150),  (1100, 700), (1300, 300), (850, 900),  
    (1500, 150), (1650, 600), (1400, 850), (1750, 450)  
]

POSICOES_PLANETAS = []
ref_w, ref_h = 1920, 1080 

for (bx, by) in POSICOES_BASE:
    novo_x = int(bx * LARGURA_TELA / ref_w)
    novo_y = int(by * ALTURA_TELA / ref_h)
    
    novo_x = max(80, min(novo_x, LARGURA_TELA - 80))
    novo_y = max(80, min(novo_y, ALTURA_TELA - 80))
    POSICOES_PLANETAS.append((novo_x, novo_y))

VELOCIDADE_MOVIMENTO = 6
DISTANCIA_GATILHO = 60 

# As 4 telas
tela_inicio = tk.Frame(janela_principal, width=LARGURA_TELA, height=ALTURA_TELA)
tela_como_jogar = tk.Frame(janela_principal, width=LARGURA_TELA, height=ALTURA_TELA)
tela_jogo = tk.Frame(janela_principal, width=LARGURA_TELA, height=ALTURA_TELA)
tela_historia = tk.Frame(janela_principal, width=LARGURA_TELA, height=ALTURA_TELA)

tela_inicio.place(x=0, y=0)
tela_como_jogar.place(x=0, y=0)
tela_jogo.place(x=0, y=0)
tela_historia.place(x=0, y=0)

# Troca telas
def troca_tela(frame_destino):
    global jogo_ligado, tempo_inicio

    tela_inicio.place_forget()
    tela_como_jogar.place_forget()
    tela_jogo.place_forget()
    tela_historia.place_forget()
    frame_destino.place(x=0, y=0)

    if frame_destino is tela_jogo:
        para_todos_os_sons() 
        
        # CORREÇÃO: Reseta o estado do jogo ao iniciar do Menu
        reseta_estado_jogo() 
        monta_tabuleiro_jogo() 
        
        jogo_ligado = True
        tempo_inicio = time.time() 
        loop_do_jogo()
    elif frame_destino in (tela_inicio, tela_como_jogar, tela_historia): 
        jogo_ligado = False
        toca_jazz_se_nao_tiver_rolando()
    else:
        jogo_ligado = False
    
    janela_principal.focus_force()

def volta_pro_menu_principal():
    global direcao_movimento, jogo_ligado
    jogo_ligado = False
    para_todos_os_sons()
    direcao_movimento = [0, 0] 
    troca_tela(tela_inicio)

# Imagens e Gráficos
def cria_imagem_vazia(tamanho, numero=None, cor_fundo=(30,30,30)):
    L, A = tamanho
    img = Image.new("RGBA", (L,A), cor_fundo + (255,))
    d = ImageDraw.Draw(img)
    if numero is not None:
        try: font = ImageFont.truetype("arial.ttf", 28)
        except: font = ImageFont.load_default()
        d.text((L/2, A/2), str(numero), fill="white", font=font, anchor="mm")
    return img

if os.path.exists(FUNDO_CAMINHO):
    img_fundo = Image.open(FUNDO_CAMINHO).resize((LARGURA_TELA, ALTURA_TELA), Image.LANCZOS)
else:
    img_fundo = cria_imagem_vazia((LARGURA_TELA, ALTURA_TELA))
fundo_tk = ImageTk.PhotoImage(img_fundo)

# Tela 1: Menu
canvas_inicio = tk.Canvas(tela_inicio, width=LARGURA_TELA, height=ALTURA_TELA, highlightthickness=0)
canvas_inicio.place(x=0, y=0)
canvas_inicio.create_image(0, 0, image=fundo_tk, anchor="nw")

canvas_inicio.create_text(LARGURA_TELA / 2, 80, text="Via Láctea da Estatística", font=("Arial", 36, "bold"), fill="white", anchor="n")
canvas_inicio.create_text(LARGURA_TELA / 2, 140, text="Responda perguntas e avance pelos planetas!", font=("Arial", 18), fill="white", anchor="n")

# Estilos de Botões
ESTILO_BOTAO_PRINCIPAL = {
    "font": ("Arial", 18, "bold"),
    "fg": "#000000",             
    "bg": "#C6E043",             
    "activebackground": "#9BCC0C", 
    "relief": tk.FLAT,           
    "padx": 20,
    "pady": 10
}

ESTILO_BOTAO_SECUNDARIO = {
    "font": ("Arial", 14),
    "fg": "#FFFFFF",             
    "bg": "#3C3F41",             
    "activebackground": "#55585A",
    "relief": tk.FLAT,
    "padx": 15,
    "pady": 5
}

# Definição e Aplicação do Hover nos Botões do Menu

botao_jogar = tk.Button(tela_inicio, text="Iniciar Missão", 
                        command=lambda: troca_tela(tela_jogo), takefocus=False, **ESTILO_BOTAO_PRINCIPAL)
botao_jogar.bind("<Enter>", on_enter_principal)
botao_jogar.bind("<Leave>", on_leave_principal)

botao_como = tk.Button(tela_inicio, text="Como Jogar", 
                       command=lambda: troca_tela(tela_como_jogar), takefocus=False, **ESTILO_BOTAO_SECUNDARIO)
botao_como.bind("<Enter>", on_enter_secundario)
botao_como.bind("<Leave>", on_leave_secundario)

botao_ranking = tk.Button(tela_inicio, text="Comandantes Estelares", 
                          command=mostra_ranking, takefocus=False, **ESTILO_BOTAO_SECUNDARIO) 
botao_ranking.bind("<Enter>", on_enter_secundario)
botao_ranking.bind("<Leave>", on_leave_secundario)

botao_historia = tk.Button(tela_inicio, text="História", 
                           command=lambda: troca_tela(tela_historia), takefocus=False, **ESTILO_BOTAO_SECUNDARIO)
botao_historia.bind("<Enter>", on_enter_secundario)
botao_historia.bind("<Leave>", on_leave_secundario)


# Posições dos botões (NOVO LAYOUT)

# 1. Botão Principal (centro)
canvas_inicio.create_window(LARGURA_TELA / 2, ALTURA_TELA * 0.50, window=botao_jogar, anchor="center")

# 2. Botões Secundários (alinhados na horizontal, abaixo)
distancia_x = 200 # Distância do centro para cada lado
pos_y_secundaria = ALTURA_TELA * 0.78 # Posição Y comum para todos

canvas_inicio.create_window(LARGURA_TELA / 2 - 250, pos_y_secundaria, window=botao_como, anchor="center")
canvas_inicio.create_window(LARGURA_TELA / 2, pos_y_secundaria, window=botao_ranking, anchor="center") 
canvas_inicio.create_window(LARGURA_TELA / 2 + 228, pos_y_secundaria, window=botao_historia, anchor="center") 


# Canvas do Jogo
def prepara_interface_jogo():
    global tela_jogo_canvas
    tela_jogo_canvas = tk.Canvas(tela_jogo, width=LARGURA_TELA, height=ALTURA_TELA, highlightthickness=0)
    tela_jogo_canvas.place(x=0, y=0)

# Nossas imagens( tamanho do planeta e foguete)
L_PLANETA, A_PLANETA = 90, 90
L_FOGUETE, A_FOGUETE = 90, 90 

lista_planetas_tk = []
for idx, pth in enumerate(CAMINHO_PLANETAS, start=1):
    if os.path.exists(pth):
        img = Image.open(pth).convert("RGBA").resize((L_PLANETA, A_PLANETA), Image.LANCZOS)
    else:
        img = cria_imagem_vazia((L_PLANETA, A_PLANETA), idx)
    lista_planetas_tk.append(ImageTk.PhotoImage(img))

nave_parada_dir = None
nave_parada_esq = None
nave_fogo_dir = None
nave_fogo_esq = None
img_do_foguete = None 

def carrega_e_prepara_foguete(caminho, largura, altura):
    if os.path.exists(caminho):
        img_original = Image.open(caminho).convert("RGBA").resize((largura, altura), Image.LANCZOS)
    else:
        img_original = cria_imagem_vazia((largura, altura))
    
    img_original.putdata([ (255, 255, 255, 0) if item[:3] == (255, 255, 255) else item for item in img_original.getdata() ])
    
    img_espelhada = img_original.transpose(Image.FLIP_LEFT_RIGHT)
    
    return ImageTk.PhotoImage(img_original), ImageTk.PhotoImage(img_espelhada)

# Carrega as 4 variações
nave_parada_dir, nave_parada_esq = carrega_e_prepara_foguete(FOGUETE_PARADO_CAMINHO, L_FOGUETE, A_FOGUETE)
nave_fogo_dir, nave_fogo_esq = carrega_e_prepara_foguete(FOGUETE_FOGO_CAMINHO, L_FOGUETE, A_FOGUETE)
img_do_foguete = nave_parada_dir

# Monta o jogo (reseta tudo)
def monta_tabuleiro_jogo():
    global id_foguete, ids_planetas, id_destaque, id_hud_vidas, id_hud_pontos, pos_x_foguete, pos_y_foguete, img_do_foguete
    
    tela_jogo_canvas.delete("all") 
    tela_jogo_canvas.create_image(0, 0, image=fundo_tk, anchor="nw")

    ids_planetas = []
    id_destaque = tela_jogo_canvas.create_rectangle(0, 0, 0, 0, outline="yellow", width=4)

    for i in range(len(lista_planetas_tk)):
        if i >= len(POSICOES_PLANETAS): break 
        px, py = POSICOES_PLANETAS[i]
        pid = tela_jogo_canvas.create_image(px, py, image=lista_planetas_tk[i], anchor="center")
        ids_planetas.append(pid)

    x_start, y_start = POSICOES_PLANETAS[0] 
    pos_x_foguete = x_start 
    pos_y_foguete = y_start - 60 
    
    img_do_foguete = nave_parada_dir
    id_foguete = tela_jogo_canvas.create_image(pos_x_foguete, pos_y_foguete, image=img_do_foguete, anchor="center")

    # HUD (Vidas e Pontos na tela)
    tela_jogo_canvas.create_text(20, 30, text="Vidas:", font=("Arial", 20, "bold"), fill="white", anchor="w")
    id_hud_vidas = tela_jogo_canvas.create_text(100, 30, text="❤❤❤", font=("Arial", 20), fill="red", anchor="w")
    
    tela_jogo_canvas.create_text(LARGURA_TELA - 200, 30, text="Pontos:", font=("Arial", 20, "bold"), fill="white", anchor="w")
    id_hud_pontos = tela_jogo_canvas.create_text(LARGURA_TELA - 90, 30, text="0", font=("Arial", 20), fill="yellow", anchor="w")

    botao_voltar = tk.Button(tela_jogo, text="Menu", font=("Arial", 12), command=volta_pro_menu_principal, takefocus=False)
    tela_jogo_canvas.create_window(LARGURA_TELA - 100, ALTURA_TELA - 50, window=botao_voltar, anchor="center")

    atualiza_visuais_hud()

def atualiza_visuais_hud():
    global id_hud_vidas, id_hud_pontos, id_destaque
    tela_jogo_canvas.itemconfig(id_hud_vidas, text="❤"*vidas)
    tela_jogo_canvas.itemconfig(id_hud_pontos, text=str(pontos))

    # Destaque no planeta alvo
    if posicao_alvo < len(POSICOES_PLANETAS):
        px, py = POSICOES_PLANETAS[posicao_alvo]
        tamanho_destaque = L_PLANETA // 2 + 5
        tela_jogo_canvas.coords(id_destaque, px - tamanho_destaque, py - tamanho_destaque, px + tamanho_destaque, py + tamanho_destaque)
    else:
        tela_jogo_canvas.coords(id_destaque, -100, -100, -90, -90)

# Telas secundárias
def configura_tela_historia():
    canvas_historia = tk.Canvas(tela_historia, width=LARGURA_TELA, height=ALTURA_TELA, highlightthickness=0)
    canvas_historia.place(x=0, y=0)
    canvas_historia.create_image(0, 0, image=fundo_tk, anchor="nw")

    titulo = "✨ Missão: Desvendando a Estatística Espacial ✨\n"
    historia_texto = ("Comandante, nossa frota precisa de você!\n\n "
        "A Via Láctea está cheia de perguntas "
        "de Estatística que ameaçam a harmonia do universo.\n"
        "Sua missão é responder as perguntas dos 10 planetas do conhecimento para chegar no planeta final.\n" 
        "Boa sorte, Comandante. O destino da galáxia está em suas mãos!"
    )
    
    canvas_historia.create_text(LARGURA_TELA / 2, 100, text=titulo, 
                                  font=("Arial", 30, "bold"), fill="#C6E043", anchor="n")
    
    canvas_historia.create_text(LARGURA_TELA / 2, ALTURA_TELA * 0.35, text=historia_texto, 
                                  font=("Arial", 18), fill="white", anchor="center", 
                                  width=LARGURA_TELA * 0.75, justify='center')

    # Botão de voltar (ESTILO PRINCIPAL)
    botao_continuar = tk.Button(tela_historia, text="Continuar para o Menu", 
                              command=lambda: troca_tela(tela_inicio), takefocus=False, **ESTILO_BOTAO_PRINCIPAL)
    botao_continuar.bind("<Enter>", on_enter_principal)
    botao_continuar.bind("<Leave>", on_leave_principal)
                              
    canvas_historia.create_window(LARGURA_TELA / 2, ALTURA_TELA * 0.8, window=botao_continuar, anchor="center")

def configura_como_jogar():
    canvas_como = tk.Canvas(tela_como_jogar, width=LARGURA_TELA, height=ALTURA_TELA, highlightthickness=0)
    canvas_como.place(x=0, y=0)
    canvas_como.create_image(0, 0, image=fundo_tk, anchor="nw") 

    como_jogar_texto = (
        "🚀 REGRAS DE NAVEGAÇÃO 🚀\n\n"
        "- Use as teclas WASD ou as SETAS para pilotar o foguete.\n"
        "- W / SETA CIMA: Mover para CIMA\n" 
        "- S / SETA BAIXO: Mover para BAIXO\n" 
        "- A / SETA ESQUERDA: Mover para ESQUERDA\n"
        "- D / SETA DIREITA: Mover para DIREITA\n\n"

        "Chegue perto de cada planeta brilhante, cuidado com as respostas erradas que podem custar vidas!\n\n"
    
    )
    
    canvas_como.create_text(LARGURA_TELA / 2, ALTURA_TELA * 0.3, text=como_jogar_texto, 
                              font=("Arial", 18), fill="white", anchor="center", 
                              width=LARGURA_TELA * 0.75, justify='left', tags="how_text")
    
    # Botão de Iniciar (ESTILO PRINCIPAL)
    botao_iniciar_missao = tk.Button(tela_como_jogar, text="Missão", 
                                  command=lambda: troca_tela(tela_jogo), takefocus=False, **ESTILO_BOTAO_PRINCIPAL)
    botao_iniciar_missao.bind("<Enter>", on_enter_principal)
    botao_iniciar_missao.bind("<Leave>", on_leave_principal)
                                  
    canvas_como.create_window(LARGURA_TELA / 2, ALTURA_TELA * 0.65, window=botao_iniciar_missao, anchor="center")

    # Botão de Voltar (ESTILO SECUNDÁRIO)
    botao_voltar_menu = tk.Button(tela_como_jogar, text="Menu Principal", 
                                  command=lambda: troca_tela(tela_inicio), takefocus=False, **ESTILO_BOTAO_SECUNDARIO)
    botao_voltar_menu.bind("<Enter>", on_enter_secundario)
    botao_voltar_menu.bind("<Leave>", on_leave_secundario)
                                  
    canvas_como.create_window(LARGURA_TELA / 2, ALTURA_TELA * 0.75, window=botao_voltar_menu, anchor="center")


# O Loop (coração do jogo)
def loop_do_jogo():
    global pos_x_foguete, pos_y_foguete, posicao_alvo, som_foguete, direcao_movimento, img_do_foguete

    if not jogo_ligado:
        return

    if vidas <= 0 or em_pergunta:
        if som_foguete: para_todos_os_sons()
        janela_principal.after(20, loop_do_jogo)
        return

    dx, dy = direcao_movimento
    movendo = (dx != 0 or dy != 0)

    # Imagem do foguete: parado ou com fogo (Ajuste para virar a nave)
    if movendo:
        if dx < 0:
            img_do_foguete = nave_fogo_esq
        else: 
            img_do_foguete = nave_fogo_dir 
    else:
        if direcao_movimento[0] < 0:
            img_do_foguete = nave_parada_esq
        else:
            img_do_foguete = nave_parada_dir 
    
    if img_do_foguete is not None and id_foguete is not None:
        tela_jogo_canvas.itemconfig(id_foguete, image=img_do_foguete)


    # Liga/desliga som_foguete
    if movendo:
        if not som_foguete and os.path.exists(FOGUETE_SOM_CAMINHO):
            try:
                winsound.PlaySound(FOGUETE_SOM_CAMINHO, winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC)
                som_foguete = True
            except: pass
    else:
        if som_foguete:
            para_todos_os_sons()
            som_foguete = False

    # Movimento
    if movendo:
        pos_x_foguete += dx * VELOCIDADE_MOVIMENTO
        pos_y_foguete += dy * VELOCIDADE_MOVIMENTO
        
        pos_x_foguete = max(L_FOGUETE/2, min(pos_x_foguete, LARGURA_TELA - L_FOGUETE/2))
        pos_y_foguete = max(A_FOGUETE/2, min(pos_y_foguete, ALTURA_TELA - A_FOGUETE/2))
        tela_jogo_canvas.coords(id_foguete, pos_x_foguete, pos_y_foguete)
        
    # Chegou no alvo? (Verifica a distância)
    if posicao_alvo < len(POSICOES_PLANETAS):
        px, py = POSICOES_PLANETAS[posicao_alvo]
        distancia = ((pos_x_foguete - px)**2 + (pos_y_foguete - py)**2)**0.5
        
        if distancia < DISTANCIA_GATILHO:
            direcao_movimento = [0, 0] 
            para_todos_os_sons() 
            checa_proximidade_e_pergunta()
            
    janela_principal.after(20, loop_do_jogo)


# ==========================================================
# FUNÇÕES DE PERGUNTA E RESPOSTA (Bloco Corrigido)
# ==========================================================

# Função auxiliar para não repetir código
# Função auxiliar para não repetir código
def avalia_resposta(acertou, mensagem_erro_complementar):
    # CORREÇÃO: Declarar todas as variáveis globais necessárias
    global posicao_alvo, pontos, vidas, em_pergunta, pos_x_foguete, pos_y_foguete, id_foguete
    
    em_pergunta = False 

    if acertou:
        # Removido messagebox.showinfo para fechar o modal e fluir na hora!
        pontos += 10
        posicao_alvo += 1
        
        # CORREÇÃO DE LOOP: MOVE O FOGUETE PARA LONGE APÓS ACERTO
        if posicao_alvo < len(POSICOES_PLANETAS):
            # Move o foguete 100 pixels para a esquerda do planeta recém-visitado
            pos_x_foguete = POSICOES_PLANETAS[posicao_alvo-1][0] - 100 
            pos_y_foguete = POSICOES_PLANETAS[posicao_alvo-1][1] - 100 
            tela_jogo_canvas.coords(id_foguete, pos_x_foguete, pos_y_foguete)
            
    else:
        vidas -= 1
        # Mantemos messagebox.showerror para o jogador saber que perdeu vida.
        messagebox.showerror("Erro!", f"Resposta incorreta!\n{mensagem_erro_complementar}", parent=janela_principal)
    
    atualiza_visuais_hud()
    
    # 🚨 CORREÇÃO DO BUG: Chamamos o modal de derrota (False) diretamente se as vidas acabarem!
    if vidas <= 0:
        mostra_modal_fim(False) # <--- AQUI ESTÁ A CHAVE!
        return

    # Se acertou e não perdeu, checa se chegou no fim (vitória)
    if acertou:
        checa_fim_do_jogo()

def checa_proximidade_e_pergunta():
    """
    Abre a janela de pergunta correta (múltipla escolha ou resposta direta).
    """
    global posicao_alvo, pontos, vidas, em_pergunta
    
    # 1. Planeta 12 (Chegada)
    if posicao_alvo == len(lista_planetas_tk): 
        checa_fim_do_jogo()
        return

    # 2. índice da lista de perguntas (planeta 1 = pergunta 0)
    indice_pergunta = posicao_alvo - 1 

    # 3. Planeta de passagem (sem pergunta)
    # Se o índice da pergunta for maior que a lista de perguntas, o planeta é só de passagem.
    if indice_pergunta >= len(perguntas):
        messagebox.showinfo("Alerta", f"Planeta {posicao_alvo}: Avançando! Este planeta não tem pergunta.", parent=janela_principal)
        pontos += 10
        posicao_alvo += 1
        
        # CORREÇÃO DE LOOP: MOVE O FOGUETE PARA LONGE
        if posicao_alvo < len(POSICOES_PLANETAS):
            global pos_x_foguete, pos_y_foguete, id_foguete
            # Move o foguete 100 pixels para a esquerda do planeta recém-visitado
            pos_x_foguete = POSICOES_PLANETAS[posicao_alvo-1][0] - 100 
            pos_y_foguete = POSICOES_PLANETAS[posicao_alvo-1][1] - 100 
            tela_jogo_canvas.coords(id_foguete, pos_x_foguete, pos_y_foguete)
        
        atualiza_visuais_hud()
        em_pergunta = False
        checa_fim_do_jogo()
        return

    # 4. Carregar pergunta
    pergunta_atual = perguntas[indice_pergunta]
    enunciado = pergunta_atual.get("pergunta", "Pergunta Indefinida")
    
    em_pergunta = True
    
    # ==========================================================
    # LÓGICA DE TRATAMENTO DA PERGUNTA (Antiga vs. Alternativas)
    # ==========================================================
    
    # >>> SE TIVER ALTERNATIVAS (Perguntas Novas) <<<
    if 'alternativas' in pergunta_atual:
        
        alternativas = pergunta_atual.get("alternativas", [])
        indice_certo = pergunta_atual.get("indice_certo", -1) 
        
        def checa_resposta_alternativas():
            nonlocal modal # Para acessar a variável modal de fora

            try:
                # O Radiobutton usa valor '1', '2', etc. Precisa converter para índice 0, 1, etc.
                escolha_jogador = int(escolha_var.get()) - 1 
            except ValueError:
                messagebox.showerror("Erro!", "Escolha inválida. Tente novamente.", parent=janela_principal)
                return

            modal.destroy()
            avalia_resposta(escolha_jogador == indice_certo, 
                           f"A correta era a opção {indice_certo + 1}: {alternativas[indice_certo]}")


        modal = tk.Toplevel(janela_principal)
        modal.title(f"Planeta {posicao_alvo}: Escolha Múltipla")
        modal.transient(janela_principal)
        modal.grab_set()
        modal.geometry("600x400") 

        tk.Label(modal, text=enunciado, font=("Arial", 16), wraplength=550, justify=tk.LEFT).pack(pady=10, padx=20)
        
        escolha_var = tk.StringVar(value="") 

        for i, alt in enumerate(alternativas):
            tk.Radiobutton(
                modal,
                text=alt,
                value=str(i + 1), 
                variable=escolha_var,
                font=("Arial", 14),
                anchor="w",
                width=60, 
            ).pack(anchor="w", padx=50, pady=2)

        tk.Button(modal, text="Confirmar Resposta", font=("Arial", 14, "bold"), command=checa_resposta_alternativas).pack(pady=20)
        
    # >>> SE NÃO TIVER ALTERNATIVAS (Perguntas Antigas) <<<
    else: 
        # Mantém a lógica antiga de resposta direta por input de texto
        resposta_correta = str(pergunta_atual.get("resposta", "")).strip()

        resposta_jogador = simpledialog.askstring(f"Planeta {posicao_alvo}", enunciado, parent=janela_principal)

        if resposta_jogador is None: 
            avalia_resposta(False, f"Não foi respondida. O certo era: {resposta_correta}") 
        else:
            acertou = resposta_jogador.strip().lower() == resposta_correta.lower()
            avalia_resposta(acertou, f"O certo era: {resposta_correta}")


# ==========================================================
# FUNÇÕES DE PERGUNTA E RESPOSTA (Fim do Bloco Corrigido)
# ==========================================================


def checa_fim_do_jogo():
    if posicao_alvo == len(lista_planetas_tk): mostra_modal_fim(True)

def mostra_modal_fim(vitoria=True):
    para_todos_os_sons()
    global direcao_movimento, pontos, tempo_inicio
    direcao_movimento = [0, 0]
    
    tempo_fim = time.time()
    tempo_segundos = int(tempo_fim - tempo_inicio)

    if vitoria or pontos > 0:
        nome = simpledialog.askstring("Recorde!", f"Parabéns! Você fez {pontos} pontos em {tempo_segundos} segundos. Digite seu nome para o ranking:", parent=janela_principal)
        if nome:
            salva_e_ordena_ranking(nome[:15].strip(), pontos, tempo_segundos) 
            
    mensagem = f"{'MISSÃO CUMPRIDA! Via Láctea Resgatada!' if vitoria else 'Game Over - O Foguete Explodiu...'}\nPontos: {pontos}\nTempo: {tempo_segundos} segundos"
    
    modal = tk.Toplevel(janela_principal)
    modal.transient(janela_principal)
    modal.grab_set()
    modal.geometry("420x220+{}+{}".format(janela_principal.winfo_x()+LARGURA_TELA//2-210, janela_principal.winfo_y()+ALTURA_TELA//2-110))
    
    tk.Label(modal, text=mensagem, font=("Arial", 16)).pack(pady=10)
    
    tk.Button(modal, text="Voltar ao Menu", command=lambda: [modal.destroy(), reseta_estado_jogo(), troca_tela(tela_inicio)]).pack(pady=3)
    tk.Button(modal, text="Tentar de Novo", command=lambda: [modal.destroy(), reinicia_jogo()]).pack(pady=3)
    tk.Button(modal, text="Sair do Jogo", command=janela_principal.destroy).pack(pady=3)

def reseta_estado_jogo():
    global posicao_alvo, vidas, pontos, em_pergunta, direcao_movimento, tempo_inicio
    posicao_alvo = 1 
    vidas = 3
    pontos = 0
    direcao_movimento = [0, 0]
    em_pergunta = False
    tempo_inicio = 0 

def reinicia_jogo():
    reseta_estado_jogo()
    monta_tabuleiro_jogo()
    global tempo_inicio
    tempo_inicio = time.time() 

# Controles
def aperta_tecla(evento):
    global direcao_movimento
    tecla = evento.keysym
    if tecla == "Escape": sair_tela_cheia(); return
    if not jogo_ligado: return 

    if em_pergunta or vidas <= 0: direcao_movimento = [0, 0]; return
    if tecla in ("Right", "d", "D"): direcao_movimento[0] = 1
    elif tecla in ("Left", "a", "A"): direcao_movimento[0] = -1
    elif tecla in ("Up", "w", "W"): direcao_movimento[1] = -1
    elif tecla in ("Down", "s", "S"): direcao_movimento[1] = 1

def solta_tecla(evento):
    global direcao_movimento
    tecla = evento.keysym
    if tecla in ("Right", "d", "D", "Left", "a", "A"): direcao_movimento[0] = 0
    elif tecla in ("Up", "w", "W", "Down", "s", "S"): direcao_movimento[1] = 0

janela_principal.bind("<KeyPress>", aperta_tecla)
janela_principal.bind("<KeyRelease>", solta_tecla)

# Começa aqui
prepara_interface_jogo()
monta_tabuleiro_jogo()
configura_tela_historia() 
configura_como_jogar() 
troca_tela(tela_historia)
janela_principal.focus_force()
janela_principal.mainloop()