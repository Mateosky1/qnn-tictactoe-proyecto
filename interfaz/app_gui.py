"""Interfaz grafica: ver jugar a la red contra el Oracle, jugar contra la red y revisar resultados.

Ejecutar desde la carpeta del proyecto:
    python interfaz/app_gui.py

Pestanas
  Red vs Oracle   la red juega contra el Oracle con 2, 4 o 10 s de computo; parametros en vivo,
                  lo que piensa la red (mapa de calor), graficas de la partida y marcador de la sesion.
  Tu vs Red       juegas tu contra la red (clic en el tablero); puedes ver lo que piensa y deshacer.
  Resultados      las 1000 partidas por enfrentamiento y las curvas del entrenamiento.

Atajos: Espacio = jugar/pausa, N = siguiente jugada, R = reiniciar, 1/2/3 = Oracle 2/4/10 s,
        H = mostrar/ocultar lo que piensa la red, U = deshacer (Tu vs Red).

La red NO usa busqueda: solo app/inferencia.py (1 forward, o 8 simetrias en un forward).
El Oracle solo se usa como rival.
"""
import json
import os
import queue
import random
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'app'))
sys.path.insert(0, os.path.join(ROOT, 'evaluacion'))

import matplotlib  # noqa: E402
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from game_rules import EMPTY, P1, P2, CELLS, BOARD_N, legal_moves, apply_move, game_over, winning_line  # noqa: E402

# ----------------------------------------------------------------------------------- paleta
BG = '#030604'        # fondo
PANEL = '#09110b'     # tarjetas
PANEL2 = '#0e1a11'    # tarjeta resaltada / hover
LINEA = '#173322'     # bordes y rejilla
NEON = '#39ff6a'      # verde principal (la red)
NEON2 = '#1fbf4f'     # verde medio
NEON_OSC = '#0f5a27'  # verde oscuro
TEXTO = '#d4f7dc'
TEXTO2 = '#7fa58a'
RIVAL = '#e9f1eb'     # fichas del rival (Oracle / humano)
ROJO = '#ff4d6d'      # derrotas
AMBAR = '#ffc94d'     # empates

MODELOS = {'Ultimo (6466 updates)': 'largo1_final', 'Primer (500 updates)': 'largo1_gen500'}
TIEMPOS = [2, 4, 10]


def mezclar(c1, c2, t):
    """Mezcla dos colores hex (t=0 -> c1, t=1 -> c2). Tk no tiene transparencia."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return '#%02x%02x%02x' % tuple(int(round(x + (y - x) * max(0.0, min(1.0, t)))) for x, y in zip(a, b))


def fuente_mono(root):
    fams = set(tkfont.families(root))
    for f in ('JetBrains Mono', 'SF Mono', 'Menlo', 'Cascadia Mono', 'Consolas', 'DejaVu Sans Mono', 'Courier New'):
        if f in fams:
            return f
    return 'TkFixedFont'


def nombre_casilla(i):
    return f'{"ABCDE"[i % BOARD_N]}{i // BOARD_N + 1}'


# ------------------------------------------------------------------------------------ motor
class Motor:
    """Carga modelos (con cache) y el Oracle. El Oracle conserva su tabla entre partidas."""

    def __init__(self):
        self._modelos = {}
        self._oracle = None
        self.lock = threading.Lock()

    def modelo(self, nombre):
        if nombre not in self._modelos:
            import torch
            torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))
            from az_resnet_model import load_checkpoint
            self._modelos[nombre], _ = load_checkpoint(os.path.join(ROOT, 'modelos', nombre + '.pt'))
        return self._modelos[nombre]

    def analizar(self, nombre, board, player, simetrias):
        from inferencia import analizar
        t0 = time.perf_counter()
        r = analizar(self.modelo(nombre), board, player, simetrias)
        r['ms'] = (time.perf_counter() - t0) * 1000
        return r

    def oracle(self):
        if self._oracle is None:
            from oracle_minimax import OracleMinimax
            self._oracle = OracleMinimax()
        return self._oracle


# ---------------------------------------------------------------------------- widgets base
class Boton(tk.Label):
    """Boton plano con hover; `toggle` lo deja marcado."""

    def __init__(self, master, texto, comando, fuente, ancho=None, primario=False):
        super().__init__(master, text=texto, font=fuente, bg=NEON if primario else PANEL2,
                         fg=BG if primario else TEXTO, padx=12, pady=6, cursor='hand2',
                         highlightthickness=1, highlightbackground=NEON if primario else LINEA)
        if ancho:
            self.config(width=ancho)
        self.comando, self.primario, self.activo, self.habil = comando, primario, False, True
        self.bind('<Button-1>', lambda e: self.habil and self.comando())
        self.bind('<Enter>', lambda e: self._pintar(True))
        self.bind('<Leave>', lambda e: self._pintar(False))

    def _pintar(self, hover=False):
        if not self.habil:
            self.config(bg=PANEL, fg='#3d5c46', highlightbackground=LINEA)
        elif self.primario or self.activo:
            self.config(bg=NEON if not hover else '#7dff9c', fg=BG, highlightbackground=NEON)
        else:
            self.config(bg=PANEL2 if not hover else NEON_OSC, fg=TEXTO,
                        highlightbackground=NEON2 if hover else LINEA)

    def marcar(self, activo):
        self.activo = activo
        self._pintar()

    def habilitar(self, h):
        self.habil = h
        self.config(cursor='hand2' if h else 'arrow')
        self._pintar()


class Selector(tk.Frame):
    """Fila de opciones excluyentes (segmented control)."""

    def __init__(self, master, etiqueta, opciones, valor, al_cambiar, f, fe):
        super().__init__(master, bg=PANEL)
        tk.Label(self, text=etiqueta, font=fe, bg=PANEL, fg=TEXTO2, width=20, anchor='w').pack(side='left')
        self.botones, self.valor, self.al_cambiar = {}, valor, al_cambiar
        for txt, val in opciones:
            b = Boton(self, txt, lambda v=val: self.set(v), f)
            b.pack(side='left', padx=(0, 4))
            self.botones[val] = b
        self.set(valor, avisar=False)

    def set(self, v, avisar=True):
        self.valor = v
        for val, b in self.botones.items():
            b.marcar(val == v)
        if avisar and self.al_cambiar:
            self.al_cambiar(v)

    def habilitar(self, h):
        for b in self.botones.values():
            b.habilitar(h)
        if h:
            self.set(self.valor, avisar=False)


class Tarjeta(tk.Frame):
    """Dato en vivo: valor grande + etiqueta. Destella al cambiar."""

    def __init__(self, master, etiqueta, fv, fe, ancho=12):
        super().__init__(master, bg=PANEL, highlightthickness=1, highlightbackground=LINEA, padx=10, pady=6)
        self.v = tk.Label(self, text='--', font=fv, bg=PANEL, fg=NEON, anchor='w', width=ancho)
        self.v.pack(anchor='w')
        tk.Label(self, text=etiqueta, font=fe, bg=PANEL, fg=TEXTO2, anchor='w').pack(anchor='w')
        self._ultimo = None

    def set(self, texto, color=NEON):
        if texto != self._ultimo:
            self._ultimo = texto
            self.v.config(text=texto, fg=color)
            self._destello(0)

    def _destello(self, k):
        pasos = 8
        if k > pasos:
            self.config(highlightbackground=LINEA)
            return
        self.config(highlightbackground=mezclar(NEON, LINEA, k / pasos))
        self.after(35, self._destello, k + 1)


# --------------------------------------------------------------------------------- tablero
class Tablero(tk.Canvas):
    """Tablero 5x5 animado: fichas que se dibujan, ultima jugada que late, linea ganadora,
    mapa de calor de la politica de la red y casilla bajo el cursor."""

    def __init__(self, master, lado, fuente, al_clic=None):
        super().__init__(master, width=lado, height=lado, bg=BG, highlightthickness=0)
        self.lado, self.m = lado, 34
        self.c = (lado - 2 * self.m) / BOARD_N
        self.f, self.al_clic = fuente, al_clic
        self.board = [EMPTY] * CELLS
        self.color_de = {P1: NEON, P2: RIVAL}
        self.probs, self.ver_calor, self.hover, self.ultimo = None, True, None, None
        self.fase = 0.0
        self._dibujar_base()
        self.bind('<Button-1>', self._clic)
        self.bind('<Motion>', self._mov)
        self.bind('<Leave>', lambda e: self._set_hover(None))
        self._latir()

    # geometria
    def xy(self, i):
        r, c = divmod(i, BOARD_N)
        return self.m + c * self.c, self.m + r * self.c

    def centro(self, i):
        x, y = self.xy(i)
        return x + self.c / 2, y + self.c / 2

    def casilla(self, x, y):
        c, r = int((x - self.m) // self.c), int((y - self.m) // self.c)
        return r * BOARD_N + c if 0 <= r < BOARD_N and 0 <= c < BOARD_N else None

    def _dibujar_base(self):
        self.delete('all')
        for i in range(CELLS):
            x, y = self.xy(i)
            self.create_rectangle(x + 3, y + 3, x + self.c - 3, y + self.c - 3, fill=PANEL, outline=LINEA,
                                  width=1, tags=('celda', f'c{i}'))
        for k in range(BOARD_N):
            x, y = self.xy(k)
            self.create_text(x + self.c / 2, self.m / 2, text='ABCDE'[k], fill=TEXTO2, font=self.f)
            x, y = self.xy(k * BOARD_N)
            self.create_text(self.m / 2, y + self.c / 2, text=str(k + 1), fill=TEXTO2, font=self.f)

    # estado
    def reiniciar(self):
        self.board = [EMPTY] * CELLS
        self.probs, self.ultimo = None, None
        self._dibujar_base()

    def set_colores(self, color_p1, color_p2):
        self.color_de = {P1: color_p1, P2: color_p2}

    def set_probs(self, probs):
        self.probs = probs
        self._pintar_celdas()

    def colocar(self, i, jugador, al_terminar=None):
        self.board[i] = jugador
        self.ultimo = i
        self.probs = None
        self._pintar_celdas()
        self._animar_ficha(i, jugador, 0, al_terminar)

    def redibujar(self, board, ultimo=None):
        """Dibuja un tablero completo sin animar (deshacer, cambios de pestana)."""
        self.reiniciar()
        self.board = list(board)
        for i, v in enumerate(board):
            if v != EMPTY:
                self._ficha(i, v, 1.0)
        self.ultimo = ultimo
        self._pintar_celdas()

    # dibujo de fichas
    def _ficha(self, i, jugador, t, tag='ficha'):
        cx, cy = self.centro(i)
        r = self.c * 0.30
        col = self.color_de[jugador]
        w = max(3, int(self.c * 0.07))
        if jugador == P1:   # X: dos trazos que crecen
            t1, t2 = min(1, t * 2), max(0, t * 2 - 1)
            self.create_line(cx - r, cy - r, cx - r + 2 * r * t1, cy - r + 2 * r * t1, fill=col, width=w,
                             capstyle='round', tags=(tag, f'f{i}'))
            if t2 > 0:
                self.create_line(cx + r, cy - r, cx + r - 2 * r * t2, cy - r + 2 * r * t2, fill=col, width=w,
                                 capstyle='round', tags=(tag, f'f{i}'))
        else:               # O: arco que se cierra
            self.create_arc(cx - r, cy - r, cx + r, cy + r, start=90, extent=-359.9 * t, style='arc',
                            outline=col, width=w, tags=(tag, f'f{i}'))

    def _animar_ficha(self, i, jugador, k, al_terminar):
        pasos = 12
        self.delete(f'f{i}')
        self.delete('onda')
        t = (k + 1) / pasos
        self._ficha(i, jugador, t)
        cx, cy = self.centro(i)
        rr = self.c * (0.3 + 0.35 * t)
        self.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline=mezclar(self.color_de[jugador], BG, t),
                         width=2, tags='onda')
        if k + 1 < pasos:
            self.after(18, self._animar_ficha, i, jugador, k + 1, al_terminar)
        else:
            self.delete('onda')
            if al_terminar:
                al_terminar()

    def linea_ganadora(self, linea, color):
        a, b = self.centro(linea[0]), self.centro(linea[-1])

        def paso(k, n=16):
            self.delete('ganadora')
            t = (k + 1) / n
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            self.create_line(a[0], a[1], x, y, fill=mezclar(color, BG, 0.65), width=int(self.c * 0.28),
                             capstyle='round', tags='ganadora')
            self.create_line(a[0], a[1], x, y, fill=color, width=max(4, int(self.c * 0.08)),
                             capstyle='round', tags='ganadora')
            if k + 1 < n:
                self.after(22, paso, k + 1)
        paso(0)

    # celdas: calor + hover + ultima jugada
    def _pintar_celdas(self):
        pmax = float(self.probs.max()) if self.probs is not None and self.probs.max() > 0 else 1.0
        for i in range(CELLS):
            fill, borde = PANEL, LINEA
            if self.board[i] == EMPTY and self.ver_calor and self.probs is not None and self.probs[i] > 0:
                fill = mezclar(PANEL, NEON2, 0.85 * (self.probs[i] / pmax) ** 0.8)
            if i == self.hover and self.board[i] == EMPTY:
                borde = NEON
            self.itemconfig(f'c{i}', fill=fill, outline=borde, width=2 if borde == NEON else 1)
        self.delete('pct')
        if self.ver_calor and self.probs is not None:
            for i in np.argsort(-self.probs)[:3]:
                if self.probs[i] >= 0.02 and self.board[i] == EMPTY:
                    x, y = self.xy(int(i))
                    self.create_text(x + self.c - 8, y + self.c - 8, text=f'{100 * self.probs[i]:.0f}%',
                                     anchor='se', fill=TEXTO, font=self.f, tags='pct')
        self.tag_raise('ficha')
        self.tag_raise('ganadora')

    def _latir(self):
        """La ultima jugada late suavemente (un unico movimiento continuo en pantalla)."""
        self.fase = (self.fase + 0.12) % (2 * np.pi)
        self.delete('latido')
        if self.ultimo is not None:
            x, y = self.xy(self.ultimo)
            t = 0.5 + 0.5 * np.sin(self.fase)
            self.create_rectangle(x + 3, y + 3, x + self.c - 3, y + self.c - 3,
                                  outline=mezclar(NEON_OSC, NEON, t), width=2, tags='latido')
            self.tag_lower('latido', 'ficha') if self.find_withtag('ficha') else None
        self.after(60, self._latir)

    def _set_hover(self, i):
        if i != self.hover:
            self.hover = i
            self._pintar_celdas()

    def _mov(self, e):
        self._set_hover(self.casilla(e.x, e.y) if self.al_clic else None)

    def _clic(self, e):
        i = self.casilla(e.x, e.y)
        if i is not None and self.al_clic:
            self.al_clic(i)


# ------------------------------------------------------------------------------- graficas
def estilo_ejes(ax, titulo):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(LINEA)
    ax.tick_params(colors=TEXTO2, labelsize=7.5)
    ax.set_title(titulo, color=TEXTO, fontsize=9, loc='left', pad=6)
    ax.grid(color=LINEA, linewidth=0.6)
    ax.set_axisbelow(True)


class Graficas:
    """Figura de matplotlib incrustada en Tk con estilo oscuro."""

    def __init__(self, master, filas, ancho, alto):
        self.fig = Figure(figsize=(ancho, alto), dpi=100, facecolor=PANEL)
        self.axs = [self.fig.add_subplot(filas, 1, k + 1) for k in range(filas)]
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)

    def widget(self):
        return self.canvas.get_tk_widget()

    def dibujar(self):
        self.fig.tight_layout(pad=1.0, h_pad=1.6)
        self.canvas.draw_idle()


# ============================================================================ pestana 1
class VistaOracle(tk.Frame):
    def __init__(self, app):
        super().__init__(app.cuerpo, bg=BG)
        self.app, self.motor = app, app.motor
        F, FB, FE, FV = app.f, app.fb, app.fe, app.fv
        self.cola = queue.Queue()
        self.partida_id = 0
        self.corriendo = self.pensando = False
        self.serie_restante = 0
        self.marcador = {t: {'V': 0, 'E': 0, 'D': 0} for t in TIEMPOS}
        self.historial = []

        izq = tk.Frame(self, bg=BG)
        izq.pack(side='left', fill='y', padx=(18, 10), pady=12)
        self.tablero = Tablero(izq, 520, F)
        self.tablero.pack()
        self.estado = tk.Label(izq, text='Elige el tiempo del Oracle y pulsa Jugar partida.', font=FB, bg=BG,
                               fg=TEXTO, anchor='w', width=52)
        self.estado.pack(fill='x', pady=(10, 4))
        self.barra = tk.Canvas(izq, width=520, height=8, bg=PANEL, highlightthickness=0)
        self.barra.pack()
        self.barra_txt = tk.Label(izq, text='', font=FE, bg=BG, fg=TEXTO2, anchor='w')
        self.barra_txt.pack(fill='x', pady=(2, 8))
        # marcador de la sesion
        mc = tk.Frame(izq, bg=PANEL, highlightthickness=1, highlightbackground=LINEA, padx=10, pady=8)
        mc.pack(fill='x')
        tk.Label(mc, text='Marcador de la sesion (red)', font=FB, bg=PANEL, fg=TEXTO).grid(row=0, column=0,
                                                                                          columnspan=4, sticky='w')
        self.lbl_marc = {}
        for k, t in enumerate(TIEMPOS):
            tk.Label(mc, text=f'Oracle {t} s', font=F, bg=PANEL, fg=TEXTO2, width=11, anchor='w').grid(
                row=k + 1, column=0, sticky='w')
            fila = {}
            for j, (clave, col) in enumerate((('V', NEON), ('E', AMBAR), ('D', ROJO))):
                l = tk.Label(mc, text=f'{clave} 0', font=F, bg=PANEL, fg=col, width=7, anchor='w')
                l.grid(row=k + 1, column=j + 1, sticky='w')
                fila[clave] = l
            self.lbl_marc[t] = fila
        # historial de jugadas
        tk.Label(izq, text='Jugadas', font=FB, bg=BG, fg=TEXTO, anchor='w').pack(fill='x', pady=(10, 2))
        self.lista = tk.Listbox(izq, font=F, bg=PANEL, fg=TEXTO, height=7, borderwidth=0, highlightthickness=1,
                                highlightbackground=LINEA, selectbackground=NEON_OSC, activestyle='none')
        self.lista.pack(fill='x')

        der = tk.Frame(self, bg=BG)
        der.pack(side='left', fill='both', expand=True, padx=(10, 18), pady=12)
        ctl = tk.Frame(der, bg=PANEL, highlightthickness=1, highlightbackground=LINEA, padx=12, pady=10)
        ctl.pack(fill='x')
        self.sel_t = Selector(ctl, 'Tiempo del Oracle', [(f'{t} s', t) for t in TIEMPOS], 2, None, F, FE)
        self.sel_t.pack(anchor='w', pady=2)
        self.sel_m = Selector(ctl, 'Checkpoint', [(k, v) for k, v in MODELOS.items()], 'largo1_final', None, F, FE)
        self.sel_m.pack(anchor='w', pady=2)
        self.sel_inf = Selector(ctl, 'Modo de la red', [('Simetrias x8', True), ('Pura (1 forward)', False)], True,
                                None, F, FE)
        self.sel_inf.pack(anchor='w', pady=2)
        self.sel_emp = Selector(ctl, 'Empieza (X)', [('Red', 'red'), ('Oracle', 'oracle'), ('Azar', 'azar')], 'azar',
                                None, F, FE)
        self.sel_emp.pack(anchor='w', pady=2)
        self.sel_ap = Selector(ctl, 'Jugadas al azar', [('0', 0), ('1', 1), ('2', 2), ('3', 3), ('1-3', -1)], -1,
                               None, F, FE)
        self.sel_ap.pack(anchor='w', pady=2)
        self.vel = Selector(ctl, 'Pausa entre jugadas', [('1 s', 1000), ('0.5 s', 500), ('0.1 s', 100), ('Sin pausa', 0)],
                            500, None, F, FE)
        self.vel.pack(anchor='w', pady=2)
        bt = tk.Frame(ctl, bg=PANEL)
        bt.pack(anchor='w', pady=(8, 0))
        self.b_jugar = Boton(bt, 'Jugar partida', self.jugar_pausa, FB, primario=True)
        self.b_jugar.pack(side='left', padx=(0, 6))
        self.b_paso = Boton(bt, 'Siguiente jugada', self.paso, F)
        self.b_paso.pack(side='left', padx=(0, 6))
        self.b_serie = Boton(bt, 'Serie de 10', lambda: self.iniciar_serie(10), F)
        self.b_serie.pack(side='left', padx=(0, 6))
        Boton(bt, 'Reiniciar', self.reiniciar, F).pack(side='left', padx=(0, 6))
        self.b_guardar = Boton(bt, 'Guardar partida', self.guardar, F)
        self.b_guardar.pack(side='left')

        tj = tk.Frame(der, bg=BG)
        tj.pack(fill='x', pady=(10, 0))
        self.t_turno = Tarjeta(tj, 'jugada', FV, FE, 7)
        self.t_eval = Tarjeta(tj, 'la red se ve ganando', FV, FE, 7)
        self.t_conf = Tarjeta(tj, 'confianza de la red', FV, FE, 7)
        self.t_onodos = Tarjeta(tj, 'nodos del Oracle', FV, FE, 8)
        self.t_oexacto = Tarjeta(tj, 'calculo del Oracle', FV, FE, 10)
        self.t_otiempo = Tarjeta(tj, 'tiempo del Oracle', FV, FE, 7)
        for k, t in enumerate((self.t_turno, self.t_eval, self.t_conf, self.t_onodos, self.t_oexacto, self.t_otiempo)):
            t.grid(row=k // 3, column=k % 3, sticky='we', padx=(0, 8), pady=(0, 8))
        for c in range(3):
            tj.columnconfigure(c, weight=1)

        self.g = Graficas(der, 3, 6.4, 4.6)
        self.g.widget().pack(fill='both', expand=True)
        self.limpiar_graficas()
        self.reiniciar()

    # ---------------------------------------------------------------- control de partida
    def reiniciar(self):
        self.partida_id += 1
        self.corriendo = self.pensando = False
        self.serie_restante = 0
        self.b_jugar.config(text='Jugar partida')
        self.tablero.reiniciar()
        self.lista.delete(0, 'end')
        self.historial, self.board, self.turno, self.fin = [], [EMPTY] * CELLS, P1, False
        self.barra.delete('all')
        self.barra_txt.config(text='')
        for t in (self.t_turno, self.t_eval, self.t_conf, self.t_onodos, self.t_oexacto, self.t_otiempo):
            t.set('--', TEXTO2)
        self.limpiar_graficas()
        self.bloquear(False)
        self.estado.config(text='Elige el tiempo del Oracle y pulsa Jugar partida.', fg=TEXTO)
        self.preparada = False

    def preparar(self):
        """Fija quien es X, aplica las jugadas al azar y bloquea los parametros."""
        self.tiempo = self.sel_t.valor
        self.modelo = self.sel_m.valor
        self.sim = self.sel_inf.valor
        emp = self.sel_emp.valor if self.sel_emp.valor != 'azar' else random.choice(['red', 'oracle'])
        self.red = P1 if emp == 'red' else P2
        self.tablero.set_colores(NEON if self.red == P1 else RIVAL, NEON if self.red == P2 else RIVAL)
        n_ap = self.sel_ap.valor if self.sel_ap.valor >= 0 else random.choice([1, 2, 3])
        for _ in range(n_ap):
            mv = random.choice(legal_moves(self.board))
            self.board = apply_move(self.board, mv, self.turno)
            self.tablero.redibujar(self.board, mv)
            self.registrar(mv, 'azar', {})
            self.turno = 3 - self.turno
        self.preparada = True
        self.bloquear(True)
        self.motor.modelo(self.modelo)
        lado = 'X (mueve primero)' if self.red == P1 else 'O'
        self.estado.config(text=f'La red juega con {lado} contra el Oracle de {self.tiempo} s.', fg=TEXTO)

    def bloquear(self, b):
        for s in (self.sel_t, self.sel_m, self.sel_inf, self.sel_emp, self.sel_ap):
            s.habilitar(not b)

    def jugar_pausa(self):
        if self.fin:
            self.reiniciar()
        if not self.preparada:
            self.preparar()
        self.corriendo = not self.corriendo
        self.b_jugar.config(text='Pausa' if self.corriendo else 'Continuar')
        if self.corriendo and not self.pensando:
            self.siguiente()

    def paso(self):
        if self.fin:
            return
        if not self.preparada:
            self.preparar()
        self.corriendo = False
        self.b_jugar.config(text='Continuar')
        if not self.pensando:
            self.siguiente()

    def iniciar_serie(self, n):
        self.reiniciar()
        self.serie_restante = n
        self.vel.set(100 if self.vel.valor > 100 else self.vel.valor, avisar=False)
        self.jugar_pausa()

    def siguiente(self):
        if self.fin or self.pensando:
            return
        done, w = game_over(self.board)
        if done:
            return self.terminar(w)
        if self.turno == self.red:
            self.jugada_red()
        else:
            self.jugada_oracle()

    def jugada_red(self):
        a = self.motor.analizar(self.modelo, self.board, self.turno, self.sim)
        self._probs_red = a['probs']
        self.tablero.set_probs(a['probs'])
        info = {'valor_red': a['valor'], 'conf': float(a['probs'].max()), 'ms': a['ms'], 'probs': a['probs']}
        self.estado.config(text=f'La red elige {nombre_casilla(a["jugada"])} '
                                f'({100 * info["conf"]:.0f}% de su politica, {a["ms"]:.0f} ms)', fg=NEON)
        pid = self.partida_id
        self.after(350 if self.vel.valor > 0 else 0, lambda: pid == self.partida_id and self.aplicar(a['jugada'], 'red', info))

    def jugada_oracle(self):
        self.pensando = True
        pid, board, turno, t = self.partida_id, list(self.board), self.turno, self.tiempo
        self.t_inicio = time.time()
        self.estado.config(text=f'El Oracle esta calculando (limite {t} s)...', fg=RIVAL)

        def trabajo():
            with self.motor.lock:
                rep = self.motor.oracle().select(board, turno, time_limit_s=t)
            self.cola.put((pid, rep))
        threading.Thread(target=trabajo, daemon=True).start()
        self.vigilar_oracle()

    def vigilar_oracle(self):
        try:
            pid, rep = self.cola.get_nowait()
        except queue.Empty:
            if self.pensando:
                el = time.time() - self.t_inicio
                self.dibujar_barra(min(1.0, el / self.tiempo), f'Oracle: {el:4.1f} / {self.tiempo} s')
                self.after(40, self.vigilar_oracle)
            return
        if pid != self.partida_id:
            return
        self.pensando = False
        ex = rep.extra
        info = {'nodos': rep.nodes, 'exacto': not ex.get('timed_out'), 'seg': rep.elapsed_ms / 1000,
                'valor_oracle': ex.get('valor_exacto'), 'raices': (ex.get('raices_resueltas'), ex.get('raices_total'))}
        self.dibujar_barra(min(1.0, info['seg'] / self.tiempo), f'Oracle: {info["seg"]:.2f} s  |  '
                           + ('resolvio la posicion de forma exacta' if info['exacto'] else
                              f'se quedo sin tiempo ({info["raices"][0]}/{info["raices"][1]} jugadas resueltas)'))
        self.aplicar(rep.move, 'oracle', info)

    def dibujar_barra(self, frac, texto):
        self.barra.delete('all')
        w = int(520 * frac)
        self.barra.create_rectangle(0, 0, w, 8, fill=NEON2 if frac < 1 else AMBAR, width=0)
        self.barra_txt.config(text=texto)

    def aplicar(self, mv, quien, info):
        jugador = self.turno
        self.board = apply_move(self.board, mv, jugador)
        self.registrar(mv, quien, info)
        self.turno = 3 - self.turno
        self.actualizar_datos()
        pid = self.partida_id

        def despues():
            if pid != self.partida_id:
                return
            done, w = game_over(self.board)
            if done:
                return self.terminar(w)
            if self.corriendo:
                self.after(int(self.vel.valor), lambda: pid == self.partida_id and self.corriendo and self.siguiente())
        self.tablero.colocar(mv, jugador, despues)

    def registrar(self, mv, quien, info):
        n = len(self.historial) + 1
        reg = {'n': n, 'quien': quien, 'jugador': 'X' if self.turno == P1 else 'O', 'casilla': mv, **{
            k: v for k, v in info.items() if k != 'probs'}}
        self.historial.append(reg)
        extra = ''
        if quien == 'red':
            extra = f'  conf {100 * info["conf"]:.0f}%  eval {info["valor_red"]:+.2f}'
        elif quien == 'oracle':
            extra = f'  {info["seg"]:.2f}s  {info["nodos"]:,} nodos  {"exacto" if info["exacto"] else "sin tiempo"}'
        etiqueta = {'red': 'RED   ', 'oracle': 'ORACLE', 'azar': 'azar  '}[quien]
        self.lista.insert('end', f'{n:>2}. {etiqueta} {reg["jugador"]} {nombre_casilla(mv)}{extra}')
        self.lista.see('end')

    # ---------------------------------------------------------------- datos y graficas
    def actualizar_datos(self):
        h = self.historial
        self.t_turno.set(str(len(h)))
        red = [r for r in h if r['quien'] == 'red']
        ora = [r for r in h if r['quien'] == 'oracle']
        if red:
            v = red[-1]['valor_red']
            self.t_eval.set(f'{50 * (v + 1):.0f}%', NEON if v > 0.2 else (ROJO if v < -0.2 else AMBAR))
            self.t_conf.set(f'{100 * red[-1]["conf"]:.0f}%')
        if ora:
            o = ora[-1]
            self.t_onodos.set(f'{o["nodos"]:,}', RIVAL)
            self.t_oexacto.set('exacto' if o['exacto'] else 'sin tiempo', NEON if o['exacto'] else AMBAR)
            self.t_otiempo.set(f'{o["seg"]:.2f} s', RIVAL)
        self.graficas()

    def limpiar_graficas(self):
        for ax, t in zip(self.g.axs, ('Como se ve la red (cabeza de valor, +1 = gana)', 'Tiempo por jugada (s)',
                                     'Politica de la red en su ultima jugada (top 3)')):
            ax.clear()
            estilo_ejes(ax, t)
        self.g.dibujar()

    def graficas(self):
        h = self.historial
        ax1, ax2, ax3 = self.g.axs
        for ax in self.g.axs:
            ax.clear()
        estilo_ejes(ax1, 'Como se ve la red (cabeza de valor, +1 = gana)')
        red = [r for r in h if r['quien'] == 'red']
        if red:
            xs = [r['n'] for r in red]
            ys = [r['valor_red'] for r in red]
            ax1.axhline(0, color=LINEA, linewidth=1)
            ax1.fill_between(xs, ys, 0, color=NEON, alpha=0.12)
            ax1.plot(xs, ys, color=NEON, linewidth=2, marker='o', markersize=4)
        ax1.set_ylim(-1.05, 1.05)
        ax1.set_xlim(0.5, max(8, len(h) + 0.5))
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        estilo_ejes(ax2, 'Tiempo por jugada (s, log): verde red, blanco Oracle exacto, ambar sin tiempo')
        for r in h:
            if r['quien'] == 'red':
                ax2.bar(r['n'], r['ms'] / 1000, color=NEON, width=0.7)
            elif r['quien'] == 'oracle':
                ax2.bar(r['n'], r['seg'], color=RIVAL if r['exacto'] else AMBAR, width=0.7)
        lim = self.tiempo if self.preparada else 2
        ax2.axhline(lim, color=ROJO, linewidth=1, linestyle='--')
        ax2.set_yscale('log')
        ax2.set_ylim(0.002, lim * 2.5)
        ax2.set_xlim(0.5, max(8, len(h) + 0.5))
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

        estilo_ejes(ax3, 'Politica de la red en su ultima jugada (top 3)')
        if red:
            probs = self._probs_red
            idx = [int(i) for i in np.argsort(-probs)[:3] if probs[i] > 0]
            ax3.barh(range(len(idx)), [100 * probs[i] for i in idx], color=[NEON] + [NEON_OSC] * 2, height=0.6)
            ax3.set_yticks(range(len(idx)), [nombre_casilla(i) for i in idx])
            ax3.invert_yaxis()
            ax3.set_xlim(0, 100)
            for k, i in enumerate(idx):
                ax3.text(100 * probs[i] + 1, k, f'{100 * probs[i]:.1f}%', va='center', color=TEXTO, fontsize=7.5)
        self.g.dibujar()

    def terminar(self, w):
        self.fin, self.corriendo = True, False
        res = 'E' if w == 0 else ('V' if w == self.red else 'D')
        self.marcador[self.tiempo][res] += 1
        for k, l in self.lbl_marc[self.tiempo].items():
            l.config(text=f'{k} {self.marcador[self.tiempo][k]}')
        if w:
            self.tablero.linea_ganadora(winning_line(self.board, w), NEON if w == self.red else ROJO)
        txt = {'V': 'Gano la red', 'E': 'Empate', 'D': 'Gano el Oracle'}[res]
        self.estado.config(text=f'{txt} en {len(self.historial)} jugadas.', fg={'V': NEON, 'E': AMBAR, 'D': ROJO}[res])
        self.b_jugar.config(text='Nueva partida')
        self.bloquear(False)
        if self.serie_restante > 1:
            self.serie_restante -= 1
            n = self.serie_restante
            self.after(900, lambda: (self.reiniciar(), setattr(self, 'serie_restante', n), self.jugar_pausa()))
        else:
            self.serie_restante = 0

    def guardar(self):
        if not self.historial:
            return
        d = os.path.join(ROOT, 'resultados', 'partidas_interfaz')
        os.makedirs(d, exist_ok=True)
        ruta = os.path.join(d, time.strftime('partida_%Y%m%d_%H%M%S.json'))
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump({'oracle_seg': getattr(self, 'tiempo', None), 'modelo': getattr(self, 'modelo', None),
                       'simetrias': getattr(self, 'sim', None), 'red_es': 'X' if getattr(self, 'red', P1) == P1 else 'O',
                       'jugadas': self.historial}, f, indent=2, ensure_ascii=False, default=float)
        self.estado.config(text=f'Partida guardada en resultados/partidas_interfaz/{os.path.basename(ruta)}', fg=NEON)


# ============================================================================ pestana 2
class VistaHumano(tk.Frame):
    def __init__(self, app):
        super().__init__(app.cuerpo, bg=BG)
        self.app, self.motor = app, app.motor
        F, FB, FE, FV = app.f, app.fb, app.fe, app.fv
        self.marcador = {'V': 0, 'E': 0, 'D': 0}

        izq = tk.Frame(self, bg=BG)
        izq.pack(side='left', fill='y', padx=(18, 10), pady=12)
        self.tablero = Tablero(izq, 520, F, al_clic=self.clic)
        self.tablero.pack()
        self.estado = tk.Label(izq, text='', font=FB, bg=BG, fg=TEXTO, anchor='w', width=52)
        self.estado.pack(fill='x', pady=(10, 4))
        tk.Label(izq, text='Haz clic en una casilla para jugar. Las casillas verdes muestran lo que la red jugaria.',
                 font=FE, bg=BG, fg=TEXTO2, anchor='w', wraplength=520, justify='left').pack(fill='x')

        der = tk.Frame(self, bg=BG)
        der.pack(side='left', fill='both', expand=True, padx=(10, 18), pady=12)
        ctl = tk.Frame(der, bg=PANEL, highlightthickness=1, highlightbackground=LINEA, padx=12, pady=10)
        ctl.pack(fill='x')
        self.sel_lado = Selector(ctl, 'Tu juegas con', [('X (primero)', P1), ('O (segundo)', P2)], P1, None, F, FE)
        self.sel_lado.pack(anchor='w', pady=2)
        self.sel_m = Selector(ctl, 'Checkpoint', [(k, v) for k, v in MODELOS.items()], 'largo1_final', None, F, FE)
        self.sel_m.pack(anchor='w', pady=2)
        self.sel_inf = Selector(ctl, 'Modo de la red', [('Simetrias x8', True), ('Pura (1 forward)', False)], True,
                                None, F, FE)
        self.sel_inf.pack(anchor='w', pady=2)
        self.sel_calor = Selector(ctl, 'Ver lo que piensa', [('Si', True), ('No', False)], True, self.toggle_calor, F, FE)
        self.sel_calor.pack(anchor='w', pady=2)
        bt = tk.Frame(ctl, bg=PANEL)
        bt.pack(anchor='w', pady=(8, 0))
        Boton(bt, 'Nueva partida', self.nueva, FB, primario=True).pack(side='left', padx=(0, 6))
        Boton(bt, 'Deshacer', self.deshacer, F).pack(side='left', padx=(0, 6))

        tj = tk.Frame(der, bg=BG)
        tj.pack(fill='x', pady=(10, 0))
        self.t_eval = Tarjeta(tj, 'la red se ve ganando', FV, FE, 7)
        self.t_sug = Tarjeta(tj, 'la red jugaria', FV, FE, 7)
        self.t_marc = Tarjeta(tj, 'tu marcador  V / E / D', FV, FE, 9)
        for k, t in enumerate((self.t_eval, self.t_sug, self.t_marc)):
            t.grid(row=0, column=k, sticky='we', padx=(0, 8))
            tj.columnconfigure(k, weight=1)
        self.g = Graficas(der, 2, 6.4, 4.4)
        self.g.widget().pack(fill='both', expand=True, pady=(10, 0))
        self.nueva()

    def toggle_calor(self, v):
        self.tablero.ver_calor = v
        self.tablero._pintar_celdas()

    def nueva(self):
        self.humano = self.sel_lado.valor
        self.red = 3 - self.humano
        self.tablero.set_colores(RIVAL if self.humano == P1 else NEON, RIVAL if self.humano == P2 else NEON)
        self.tablero.reiniciar()
        self.board, self.turno, self.fin, self.pila, self.evals = [EMPTY] * CELLS, P1, False, [], []
        self.ocupado = False
        self.t_marc.set(f'{self.marcador["V"]} / {self.marcador["E"]} / {self.marcador["D"]}', TEXTO)
        self.estado.config(text='Tu turno.' if self.humano == P1 else 'La red abre la partida...', fg=TEXTO)
        if self.red == P1:
            self.after(400, self.mueve_red)
        else:
            self.pensar()

    def pensar(self):
        """Muestra lo que la red jugaria en tu lugar (pista) y su evaluacion."""
        a = self.motor.analizar(self.sel_m.valor, self.board, self.turno, self.sel_inf.valor)
        self.tablero.set_probs(a['probs'])
        self.t_sug.set(nombre_casilla(a['jugada']) if a['jugada'] is not None else '--', TEXTO)
        v = -a['valor']   # valor desde el punto de vista de la red
        self.t_eval.set(f'{50 * (v + 1):.0f}%', NEON if v > 0.2 else (ROJO if v < -0.2 else AMBAR))
        self.evals.append((len(self.pila) + 1, v))
        self.graficas(a['probs'])

    def clic(self, i):
        if self.fin or self.ocupado or self.turno != self.humano or self.board[i] != EMPTY:
            return
        self.ocupado = True
        self.pila.append(list(self.board))
        self.board = apply_move(self.board, i, self.humano)
        self.turno = self.red

        def luego():
            if not self.revisar():
                self.estado.config(text='La red esta pensando...', fg=NEON)
                self.after(250, self.mueve_red)
            else:
                self.ocupado = False
        self.tablero.colocar(i, self.humano, luego)

    def mueve_red(self):
        a = self.motor.analizar(self.sel_m.valor, self.board, self.turno, self.sel_inf.valor)
        self.pila.append(list(self.board))
        mv = a['jugada']
        self.board = apply_move(self.board, mv, self.red)
        self.turno = self.humano
        self.evals.append((len(self.pila), a['valor']))

        def luego():
            self.ocupado = False
            if not self.revisar():
                self.estado.config(text=f'La red jugo {nombre_casilla(mv)} ({100 * a["probs"].max():.0f}% de su politica). '
                                        'Tu turno.', fg=TEXTO)
                self.pensar()
        self.tablero.colocar(mv, self.red, luego)

    def revisar(self):
        done, w = game_over(self.board)
        if not done:
            return False
        self.fin = True
        res = 'E' if w == 0 else ('V' if w == self.humano else 'D')
        self.marcador[res] += 1
        self.t_marc.set(f'{self.marcador["V"]} / {self.marcador["E"]} / {self.marcador["D"]}', TEXTO)
        if w:
            self.tablero.linea_ganadora(winning_line(self.board, w), NEON if w == self.red else RIVAL)
        self.tablero.set_probs(None)
        self.estado.config(text={'V': 'Ganaste a la red.', 'E': 'Empate.', 'D': 'Gano la red.'}[res],
                           fg={'V': RIVAL, 'E': AMBAR, 'D': NEON}[res])
        return True

    def deshacer(self):
        if self.ocupado or not self.pila:
            return
        # vuelve hasta que sea tu turno otra vez
        while self.pila:
            self.board = self.pila.pop()
            n = sum(v != EMPTY for v in self.board)
            self.turno = P1 if n % 2 == 0 else P2
            if self.turno == self.humano:
                break
        self.fin = False
        self.evals = [e for e in self.evals if e[0] <= len(self.pila)]
        self.tablero.redibujar(self.board)
        self.estado.config(text='Jugada deshecha. Tu turno.', fg=TEXTO)
        if self.turno == self.humano:
            self.pensar()
        else:
            self.after(300, self.mueve_red)

    def graficas(self, probs):
        ax1, ax2 = self.g.axs
        ax1.clear()
        ax2.clear()
        estilo_ejes(ax1, 'Como se ve la red durante la partida (+1 = la red gana)')
        if self.evals:
            xs, ys = zip(*self.evals)
            ax1.axhline(0, color=LINEA, linewidth=1)
            ax1.fill_between(xs, ys, 0, color=NEON, alpha=0.12)
            ax1.plot(xs, ys, color=NEON, linewidth=2, marker='o', markersize=4)
        ax1.set_ylim(-1.05, 1.05)
        ax1.set_xlim(0.5, max(8, len(self.evals) + 1.5))
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_xlabel('jugada', color=TEXTO2, fontsize=8)
        estilo_ejes(ax2, 'Lo que jugaria la red en tu posicion (top 5)')
        idx = [int(i) for i in np.argsort(-probs)[:5] if probs[i] > 0]
        ax2.barh(range(len(idx)), [100 * probs[i] for i in idx], color=[NEON] + [NEON_OSC] * 4)
        ax2.set_yticks(range(len(idx)), [nombre_casilla(i) for i in idx])
        ax2.invert_yaxis()
        ax2.set_xlim(0, 100)
        for k, i in enumerate(idx):
            ax2.text(100 * probs[i] + 1, k, f'{100 * probs[i]:.1f}%', va='center', color=TEXTO, fontsize=7.5)
        self.g.dibujar()


# ============================================================================ pestana 3
class VistaResultados(tk.Frame):
    def __init__(self, app):
        super().__init__(app.cuerpo, bg=BG)
        F, FB, FE, FV = app.f, app.fb, app.fe, app.fv
        res_p = os.path.join(ROOT, 'resultados', 'partidas', 'resumen.json')
        self.res = json.load(open(res_p, encoding='utf-8')) if os.path.exists(res_p) else {}
        man_p = os.path.join(ROOT, 'resultados', 'entrenamiento', 'purity_manifest.json')
        man = json.load(open(man_p, encoding='utf-8')) if os.path.exists(man_p) else {'args': {}}
        A = man.get('args', {})

        top = tk.Frame(self, bg=BG)
        top.pack(fill='x', padx=18, pady=(12, 6))
        datos = [('6466', 'updates de entrenamiento'), (f'{A.get("channels", 64)} x {A.get("blocks", 5)}', 'canales x bloques'),
                 ('392 126', 'parametros'), (f'{A.get("games", 512)}', 'partidas por update'),
                 (f'{A.get("minutes", 90):g} min', 'entrenamiento en GPU')]
        for k, (v, e) in enumerate(datos):
            t = Tarjeta(top, e, FV, FE, 9)
            t.grid(row=0, column=k, sticky='we', padx=(0, 8))
            t.set(v, NEON)
            top.columnconfigure(k, weight=1)
        fila = tk.Frame(self, bg=BG)
        fila.pack(fill='x', padx=18, pady=(0, 6))
        for k, t in enumerate(TIEMPOS):
            r = self.res.get(f'largo1_final__simetrias__t{t}s')
            tj = Tarjeta(fila, f'derrotas del ultimo checkpoint vs Oracle {t} s', FV, FE, 9)
            tj.grid(row=0, column=k, sticky='we', padx=(0, 8))
            fila.columnconfigure(k, weight=1)
            if r:
                tj.set(f'{r["pct_derrotas"]:.1f}%', NEON)
            else:
                tj.set('sin datos', TEXTO2)

        self.g = Graficas(self, 1, 12.6, 5.6)
        self.g.fig.clf()
        gs = self.g.fig.add_gridspec(2, 2)
        self.ax = [self.g.fig.add_subplot(gs[0, 0]), self.g.fig.add_subplot(gs[0, 1]),
                   self.g.fig.add_subplot(gs[1, 0]), self.g.fig.add_subplot(gs[1, 1])]
        self.g.widget().pack(fill='both', expand=True, padx=18, pady=(4, 12))
        self.dibujar()

    def dibujar(self):
        a1, a2, a3, a4 = self.ax
        estilo_ejes(a1, 'Resultados en 1000 partidas por enfrentamiento (V / E / D %)')
        filas = []
        for t in TIEMPOS:
            for m, et in (('largo1_gen500', 'primer'), ('largo1_final', 'ultimo')):
                r = self.res.get(f'{m}__simetrias__t{t}s')
                if r:
                    filas.append((f'{t}s {et}', r))
        for i, (et, r) in enumerate(filas):
            x = 0
            for v, c in ((r['pct_victorias'], NEON2), (r['pct_empates'], AMBAR), (r['pct_derrotas'], ROJO)):
                a1.barh(i, v, left=x, color=c, height=0.65, edgecolor=PANEL, linewidth=1.5)
                if v > 6:
                    a1.text(x + v / 2, i, f'{v:.1f}', ha='center', va='center', color=BG, fontsize=7.5, fontweight='bold')
                elif c == ROJO:
                    a1.text(x + v + 1, i, f'{v:.1f}', ha='left', va='center', color=ROJO, fontsize=7.5, fontweight='bold')
                x += v
        a1.set_yticks(range(len(filas)), [f[0] for f in filas])
        a1.invert_yaxis()
        a1.set_xlim(0, 108)
        a1.set_xticks(range(0, 101, 20))
        if not filas:
            a1.text(0.5, 0.5, 'Corre evaluacion/partidas_vs_oracle.py', transform=a1.transAxes, ha='center',
                    color=TEXTO2)
        estilo_ejes(a2, 'Derrotas con IC 95 %: primer (gris) vs ultimo (verde)')
        for j, (m, col) in enumerate((('largo1_gen500', TEXTO2), ('largo1_final', NEON))):
            for k, t in enumerate(TIEMPOS):
                r = self.res.get(f'{m}__simetrias__t{t}s')
                if r:
                    y, lo, hi = r['pct_derrotas'], r['ic95_derrotas'][0], r['ic95_derrotas'][1]
                    x = k + (j - 0.5) * 0.25
                    a2.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt='o', color=col, capsize=4, markersize=6)
                    a2.text(x + 0.07, y, f'{y:.1f}%', va='center', color=TEXTO, fontsize=7.5)
        a2.set_xticks(range(len(TIEMPOS)), [f'Oracle {t} s' for t in TIEMPOS])
        a2.set_ylim(bottom=0)
        filas_log = self.leer_log()
        if filas_log:
            u = np.array([f['update'] for f in filas_log])
            val = np.array([f['val'] for f in filas_log])
            emp = 100 * np.array([f['selfplay_draw'] for f in filas_log])
            estilo_ejes(a3, 'Funcion de perdida: termino de valor')
            a3.plot(u, val, color=NEON, alpha=0.25, linewidth=1)
            a3.plot(u, self.suave(val), color=NEON, linewidth=2)
            a3.set_ylim(0, np.percentile(val, 99.5) * 1.1)
            estilo_ejes(a4, 'Empates en self-play (%) durante el entrenamiento')
            a4.plot(u, emp, color=NEON, alpha=0.25, linewidth=1)
            a4.plot(u, self.suave(emp), color=NEON, linewidth=2)
        self.g.fig.tight_layout(pad=1.2, h_pad=2.0, w_pad=2.0)
        self.g.canvas.draw_idle()

    @staticmethod
    def suave(y, w=25):
        h = w // 2
        return np.array([np.median(y[max(0, i - h):i + h + 1]) for i in range(len(y))])

    @staticmethod
    def leer_log():
        import re
        p = os.path.join(ROOT, 'resultados', 'entrenamiento', 'log_largo1.txt')
        if not os.path.exists(p):
            return []
        raw = open(p, 'rb').read()
        txt = raw.decode('utf-16') if b'\x00' in raw[:200] else raw.decode('utf-8')
        out = []
        for linea in txt.splitlines():
            m = re.match(r'^\[(\d+)\]\s+(.*)$', linea.strip())
            if m:
                d = {'update': int(m.group(1))}
                d.update({k: float(v) for k, v in re.findall(r'(\w+)=([-+0-9.eE]+)', m.group(2))})
                out.append(d)
        return out


# ===================================================================================== app
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('QNN 4 en raya: la red contra el Oracle')
        self.configure(bg=BG)
        self.geometry('1320x900')
        self.minsize(1180, 820)
        mono = fuente_mono(self)
        self.f = (mono, 10)
        self.fb = (mono, 11, 'bold')
        self.fe = (mono, 9)
        self.fv = (mono, 18, 'bold')
        self.motor = Motor()

        cab = tk.Frame(self, bg=BG)
        cab.pack(fill='x', padx=18, pady=(14, 4))
        self.titulo = tk.Label(cab, text='', font=(mono, 20, 'bold'), bg=BG, fg=NEON)
        self.titulo.pack(side='left')
        self.tabs = {}
        tf = tk.Frame(cab, bg=BG)
        tf.pack(side='right')
        for clave, txt in (('oracle', 'Red vs Oracle'), ('humano', 'Tu vs Red'), ('resultados', 'Resultados')):
            b = Boton(tf, txt, lambda c=clave: self.mostrar(c), self.fb)
            b.pack(side='left', padx=(6, 0))
            self.tabs[clave] = b
        tk.Frame(self, bg=LINEA, height=1).pack(fill='x', padx=18, pady=(6, 0))
        self.cuerpo = tk.Frame(self, bg=BG)
        self.cuerpo.pack(fill='both', expand=True)
        self.vistas = {}
        self.estado_carga = tk.Label(self.cuerpo, text='Cargando la red...', font=self.fb, bg=BG, fg=NEON)
        self.estado_carga.pack(pady=200)
        self.after(50, self._escribir_titulo, 0)
        self.after(120, self._construir)

        self.bind('<space>', lambda e: self._atajo('jugar'))
        self.bind('<n>', lambda e: self._atajo('paso'))
        self.bind('<r>', lambda e: self._atajo('reiniciar'))
        self.bind('<u>', lambda e: self._atajo('deshacer'))
        self.bind('<h>', lambda e: self._atajo('calor'))
        for k, t in zip('123', TIEMPOS):
            self.bind(k, lambda e, t=t: self._atajo('tiempo', t))

    def _escribir_titulo(self, k):
        """Unica animacion de entrada: el titulo se escribe como en una terminal."""
        texto = 'QNN // 4 en raya'
        self.titulo.config(text=texto[:k] + ('_' if k < len(texto) or (k // 3) % 2 else ' '))
        if k < len(texto) + 12:
            self.after(55 if k < len(texto) else 250, self._escribir_titulo, k + 1)
        else:
            self.titulo.config(text=texto)

    def _construir(self):
        try:
            self.motor.modelo('largo1_final')
        except Exception as e:  # noqa: BLE001
            messagebox.showerror('No se pudo cargar la red', f'{e}\n\nRevisa que exista modelos/largo1_final.pt')
        self.estado_carga.destroy()
        self.vistas = {'oracle': VistaOracle(self), 'humano': VistaHumano(self), 'resultados': VistaResultados(self)}
        self.mostrar('oracle')

    def mostrar(self, clave):
        if not self.vistas:
            return
        for k, v in self.vistas.items():
            v.pack_forget()
            self.tabs[k].marcar(k == clave)
        self.vistas[clave].pack(fill='both', expand=True)
        self.actual = clave

    def _atajo(self, accion, arg=None):
        v = self.vistas.get(getattr(self, 'actual', None))
        if isinstance(v, VistaOracle):
            if accion == 'jugar':
                v.jugar_pausa()
            elif accion == 'paso':
                v.paso()
            elif accion == 'reiniciar':
                v.reiniciar()
            elif accion == 'tiempo' and not v.preparada:
                v.sel_t.set(arg)
            elif accion == 'calor':
                v.tablero.ver_calor = not v.tablero.ver_calor
                v.tablero._pintar_celdas()
        elif isinstance(v, VistaHumano):
            if accion == 'deshacer':
                v.deshacer()
            elif accion == 'reiniciar':
                v.nueva()
            elif accion == 'calor':
                v.sel_calor.set(not v.sel_calor.valor)


if __name__ == '__main__':
    App().mainloop()
