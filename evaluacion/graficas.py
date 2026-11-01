"""Genera todas las graficas del proyecto en resultados/graficas/.

Entrenamiento (salen del log y del historial del entrenamiento de largo1):
  01_funcion_de_perdida.png      perdida total de PPO y sus tres terminos
  02_progreso_entrenamiento.png  empates en self-play, KL, fraccion recortada y learning rate
Partidas contra el Oracle (salen de resultados/partidas/resumen.json):
  03_resumen_vs_oracle.png       V/E/D de cada enfrentamiento (primer y ultimo checkpoint x 2/4/10 s)
  04_enfrentamiento_t<N>s.png    una grafica por limite de tiempo: primer vs ultimo checkpoint
  05_derrotas_ic95.png           % de derrotas con intervalo de confianza del 95 %
  06_oracle_exacto.png           % de jugadas que el Oracle resolvio de forma exacta

Uso (desde evaluacion/):
    python graficas.py
    python graficas.py --log ../resultados/entrenamiento/log_largo1.txt --modo simetrias
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))

# Paleta (validada para daltonismo). Texto siempre en tinta neutra, nunca en el color de la serie.
SUP, TINTA, TINTA2, REJILLA = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
C_PRIMERO, C_ULTIMO = '#eb6834', '#2a78d6'          # primer checkpoint / ultimo checkpoint
C_V, C_E, C_D = '#008300', '#eda100', '#e34948'      # victorias / empates / derrotas
C_LINEA = '#2a78d6'

plt.rcParams.update({
    'figure.facecolor': SUP, 'axes.facecolor': SUP, 'savefig.facecolor': SUP,
    'axes.edgecolor': REJILLA, 'axes.labelcolor': TINTA2, 'text.color': TINTA,
    'xtick.color': TINTA2, 'ytick.color': TINTA2, 'axes.grid': True, 'grid.color': REJILLA,
    'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
})


# ------------------------------------------------------------------------------ lectura
def leer_log(ruta):
    """Lee el log del entrenamiento (UTF-8 o UTF-16, como lo deja PowerShell)."""
    raw = open(ruta, 'rb').read()
    txt = raw.decode('utf-16') if raw[:2] in (b'\xff\xfe', b'\xfe\xff') or b'\x00' in raw[:200] else raw.decode('utf-8')
    filas = []
    pat = re.compile(r'^\[(\d+)\]\s+(.*)$')
    for linea in txt.splitlines():
        m = pat.match(linea.strip())
        if not m:
            continue
        d = {'update': int(m.group(1))}
        for k, v in re.findall(r'(\w+)=([-+0-9.eE]+)s?', m.group(2)):
            try:
                d[k] = float(v)
            except ValueError:
                pass
        filas.append(d)
    return filas


def mediana_movil(y, w):
    """Mediana movil centrada: suaviza sin que un pico aislado arrastre la curva."""
    y = np.asarray(y, float)
    h = w // 2
    return np.array([np.median(y[max(0, i - h):i + h + 1]) for i in range(len(y))])


def serie(ax, x, y, titulo, ylabel, w=25):
    y = np.asarray(y, float)
    ax.plot(x, y, color=C_LINEA, alpha=0.25, linewidth=1)
    ax.plot(x, mediana_movil(y, w), color=C_LINEA, linewidth=2)
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    # escala robusta: un minibatch atipico no aplasta el resto de la curva
    lo, hi = np.percentile(y, [0.5, 99.5])
    pad = 0.08 * (hi - lo or 1)
    ax.set_ylim(lo - pad, hi + pad)
    fuera = int(((y < lo - pad) | (y > hi + pad)).sum())
    if fuera:
        ax.text(0.99, 0.95, f'{fuera} valor(es) atipico(s) fuera de escala', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color=TINTA2)


# ------------------------------------------------------------------- graficas de entrenamiento
def graf_perdida(filas, args, out):
    u = np.array([f['update'] for f in filas])
    pol = np.array([f['pol'] for f in filas])
    val = np.array([f['val'] for f in filas])
    ent = np.array([f['H'] for f in filas])
    frac = np.array([f.get('frac', 0) for f in filas])
    e0, e1 = args.get('entropy_start', 0.03), args.get('entropy_end', 0.003)
    vc = args.get('value_coef', 0.5)
    coef_ent = e0 + (e1 - e0) * frac
    total = pol + vc * val - coef_ent * ent

    fig, axs = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    serie(axs[0], u, total, f'Funcion de perdida total  L = L_politica + {vc:g}*L_valor - c_ent*H', 'perdida')
    serie(axs[1], u, pol, 'Perdida de politica (objetivo recortado de PPO)', 'perdida')
    serie(axs[2], u, val, 'Perdida de valor (error cuadratico, con recorte)', 'perdida')
    serie(axs[3], u, ent, 'Entropia de la politica (exploracion)', 'nats')
    axs[3].set_xlabel('update de entrenamiento')
    fig.text(0.01, 0.005, 'Linea clara: valor del ultimo minibatch, cada 10 updates.  Linea gruesa: mediana movil de 25 puntos (250 updates).',
             color=TINTA2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def graf_progreso(filas, hist, out):
    u = np.array([f['update'] for f in filas])
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    serie(axs[0, 0], u, 100 * np.array([f['selfplay_draw'] for f in filas]),
          'Empates en self-play (la red contra si misma)', '% de partidas')
    serie(axs[0, 1], u, [f['kl'] for f in filas], 'Divergencia KL por update (PPO)', 'KL')
    serie(axs[1, 0], u, [f['clip'] for f in filas], 'Fraccion de muestras recortadas (PPO)', 'fraccion')
    axs[1, 1].plot(u, [f['lr'] for f in filas], color=C_LINEA, linewidth=2)
    axs[1, 1].set_title('Learning rate (decaimiento coseno)')
    axs[1, 1].set_ylabel('lr')
    for ax in axs[1]:
        ax.set_xlabel('update de entrenamiento')
    if hist:
        hu = [h['update'] for h in hist]
        vr = [100 * h['vs_random'][0] / max(1, sum(h['vs_random'])) for h in hist]
        txt = (f'Control: vs jugador aleatorio la red gano {min(vr):.0f}-{max(vr):.0f} % en las '
               f'{len(hu)} evaluaciones (updates {hu[0]}-{hu[-1]}).')
        fig.text(0.01, 0.005, txt, color=TINTA2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ graficas contra el Oracle
def nombre_corto(modelo, primero, ultimo):
    if modelo == primero:
        return f'Primer checkpoint\n({modelo})'
    if modelo == ultimo:
        return f'Ultimo checkpoint\n({modelo})'
    return modelo


def etiqueta_segmento(ax, x0, ancho, y, pct, ultimo=False):
    if pct >= 6:
        ax.text(x0 + ancho / 2, y, f'{pct:.1f}%', ha='center', va='center', color='white',
                fontsize=9, fontweight='bold')
    elif ultimo:   # segmento angosto al final de la barra: etiqueta afuera, a la derecha
        ax.text(x0 + ancho + 1, y, f'{pct:.1f}%', ha='left', va='center', color=TINTA, fontsize=9,
                fontweight='bold')


def graf_resumen(res, modelos, tiempos, primero, ultimo, modo, out):
    filas = [(m, t) for t in tiempos for m in modelos if f'{m}__{modo}__t{t:g}s' in res]
    if not filas:
        return False
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(filas) + 1.8))
    etiquetas = []
    for i, (m, t) in enumerate(filas):
        r = res[f'{m}__{modo}__t{t:g}s']
        x = 0
        segs = ((r['pct_victorias'], C_V), (r['pct_empates'], C_E), (r['pct_derrotas'], C_D))
        for k, (pct, col) in enumerate(segs):
            ax.barh(i, pct, left=x, color=col, height=0.62, edgecolor=SUP, linewidth=2)
            etiqueta_segmento(ax, x, pct, i, pct, ultimo=(k == len(segs) - 1))
            x += pct
        etiquetas.append(f'Oracle {t:g} s  |  {"primer" if m == primero else "ultimo" if m == ultimo else m}'
                         f'  (n={r["partidas"]})')
    ax.set_yticks(range(len(filas)), etiquetas)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel('% de partidas')
    ax.grid(axis='y', visible=False)
    ax.set_title(f'Red vs Oracle: victorias / empates / derrotas  (modo de la red: {modo})')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_V, label='Victorias'), Patch(color=C_E, label='Empates'),
                       Patch(color=C_D, label='Derrotas')], loc='lower center',
              bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax.set_title(ax.get_title(loc='left'), loc='left', pad=28)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def graf_por_tiempo(res, modelos, t, primero, ultimo, modo, out):
    pres = [m for m in modelos if f'{m}__{modo}__t{t:g}s' in res]
    if not pres:
        return False
    cats = ['Victorias', 'Empates', 'Derrotas']
    claves = ['pct_victorias', 'pct_empates', 'pct_derrotas']
    fig, ax = plt.subplots(figsize=(8, 4.6))
    w = 0.8 / len(pres)
    for j, m in enumerate(pres):
        r = res[f'{m}__{modo}__t{t:g}s']
        col = C_PRIMERO if m == primero else C_ULTIMO
        xs = np.arange(3) + (j - (len(pres) - 1) / 2) * w
        vals = [r[k] for k in claves]
        ax.bar(xs, vals, width=w * 0.92, color=col, label=nombre_corto(m, primero, ultimo).replace('\n', ' ')
               + f'  n={r["partidas"]}')
        for x, v in zip(xs, vals):
            ax.text(x, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9, color=TINTA)
    ax.set_xticks(range(3), cats)
    ax.set_ylim(0, 105)
    ax.set_ylabel('% de partidas')
    ax.grid(axis='x', visible=False)
    ax.set_title(f'Enfrentamiento contra el Oracle con {t:g} s por jugada')
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def graf_ic(res, modelos, tiempos, primero, ultimo, modo, campo, ic, titulo, out):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    hay = False
    for j, m in enumerate(modelos):
        ts = [t for t in tiempos if f'{m}__{modo}__t{t:g}s' in res]
        if not ts:
            continue
        hay = True
        rs = [res[f'{m}__{modo}__t{t:g}s'] for t in ts]
        y = np.array([r[campo] for r in rs])
        lo = y - np.array([r[ic][0] for r in rs])
        hi = np.array([r[ic][1] for r in rs]) - y
        xs = np.arange(len(ts)) + (j - (len(modelos) - 1) / 2) * 0.18
        col = C_PRIMERO if m == primero else C_ULTIMO
        ax.errorbar(xs, y, yerr=[lo, hi], fmt='o', color=col, markersize=8, capsize=5, linewidth=2,
                    label=nombre_corto(m, primero, ultimo).replace('\n', ' '))
        for x, v in zip(xs, y):
            ax.text(x + 0.06, v, f'{v:.1f}%', va='center', fontsize=9, color=TINTA)
        ax.set_xticks(range(len(ts)), [f'Oracle {t:g} s' for t in ts])
    if not hay:
        plt.close(fig)
        return False
    ax.set_ylim(bottom=0)
    ax.set_ylabel('% de partidas')
    ax.grid(axis='x', visible=False)
    ax.set_title(titulo)
    ax.legend(frameon=False)
    fig.text(0.01, 0.005, 'Barras: intervalo de confianza del 95 % (Wilson).', color=TINTA2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def graf_oracle_exacto(res, modelos, tiempos, primero, ultimo, modo, out):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.8 / len(modelos)
    hay = False
    for j, m in enumerate(modelos):
        ts = [t for t in tiempos if f'{m}__{modo}__t{t:g}s' in res]
        if not ts:
            continue
        hay = True
        y = [res[f'{m}__{modo}__t{t:g}s']['pct_jugadas_oracle_exactas'] or 0 for t in ts]
        xs = np.array([tiempos.index(t) for t in ts]) + (j - (len(modelos) - 1) / 2) * w
        col = C_PRIMERO if m == primero else C_ULTIMO
        ax.bar(xs, y, width=w * 0.92, color=col, label=nombre_corto(m, primero, ultimo).replace('\n', ' '))
        for x, v in zip(xs, y):
            ax.text(x, v + 1, f'{v:.0f}%', ha='center', va='bottom', fontsize=9, color=TINTA)
    if not hay:
        plt.close(fig)
        return False
    ax.set_xticks(range(len(tiempos)), [f'Oracle {t:g} s' for t in tiempos])
    ax.set_ylim(0, 105)
    ax.set_ylabel('% de jugadas del Oracle')
    ax.grid(axis='x', visible=False)
    ax.set_title('Jugadas que el Oracle alcanzo a resolver de forma exacta')
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


# ------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=os.path.join(ROOT, 'resultados', 'entrenamiento', 'log_largo1.txt'))
    ap.add_argument('--historial', default=os.path.join(ROOT, 'resultados', 'entrenamiento', 'history.json'))
    ap.add_argument('--manifiesto', default=os.path.join(ROOT, 'resultados', 'entrenamiento', 'purity_manifest.json'))
    ap.add_argument('--resumen', default=os.path.join(ROOT, 'resultados', 'partidas', 'resumen.json'))
    ap.add_argument('--modelos', nargs='+', default=['largo1_gen500', 'largo1_final'])
    ap.add_argument('--tiempos', nargs='+', type=float, default=[2, 4, 10])
    ap.add_argument('--modo', default='simetrias', choices=['simetrias', 'pura'])
    ap.add_argument('--salida', default=os.path.join(ROOT, 'resultados', 'graficas'))
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)
    hechas = []

    filas = leer_log(a.log)
    args = json.load(open(a.manifiesto, encoding='utf-8')).get('args', {}) if os.path.exists(a.manifiesto) else {}
    hist = json.load(open(a.historial, encoding='utf-8')) if os.path.exists(a.historial) else []
    if filas:
        p = os.path.join(a.salida, '01_funcion_de_perdida.png'); graf_perdida(filas, args, p); hechas.append(p)
        p = os.path.join(a.salida, '02_progreso_entrenamiento.png'); graf_progreso(filas, hist, p); hechas.append(p)

    if os.path.exists(a.resumen):
        res = json.load(open(a.resumen, encoding='utf-8'))
        pri, ult = a.modelos[0], a.modelos[-1]
        p = os.path.join(a.salida, '03_resumen_vs_oracle.png')
        if graf_resumen(res, a.modelos, a.tiempos, pri, ult, a.modo, p): hechas.append(p)
        for t in a.tiempos:
            p = os.path.join(a.salida, f'04_enfrentamiento_t{t:g}s.png')
            if graf_por_tiempo(res, a.modelos, t, pri, ult, a.modo, p): hechas.append(p)
        p = os.path.join(a.salida, '05_derrotas_ic95.png')
        if graf_ic(res, a.modelos, a.tiempos, pri, ult, a.modo, 'pct_derrotas', 'ic95_derrotas',
                   'Derrotas de la red contra el Oracle', p): hechas.append(p)
        p = os.path.join(a.salida, '06_oracle_exacto.png')
        if graf_oracle_exacto(res, a.modelos, a.tiempos, pri, ult, a.modo, p): hechas.append(p)
    else:
        print('Aun no hay resultados de partidas (corre primero partidas_vs_oracle.py).')

    print('Graficas generadas:')
    for p in hechas:
        print('  ', os.path.relpath(p, ROOT))


if __name__ == '__main__':
    main()
