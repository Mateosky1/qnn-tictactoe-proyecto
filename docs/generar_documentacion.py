"""Genera docs/DOCUMENTACION.docx con la documentacion del codigo, las graficas y los resultados.

Se vuelve a correr despues de las partidas contra el Oracle para que el documento incluya los
numeros reales (lee resultados/partidas/resumen.json y resultados/graficas/*.png).

Uso (desde docs/):   python generar_documentacion.py
Requiere:            pip install python-docx
"""
import json
import os
import datetime

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
GRAF = os.path.join(ROOT, 'resultados', 'graficas')
RES = os.path.join(ROOT, 'resultados', 'partidas', 'resumen.json')
MANIF = os.path.join(ROOT, 'resultados', 'entrenamiento', 'purity_manifest.json')
MODELOS = ['largo1_gen500', 'largo1_final']
TIEMPOS = [2, 4, 10]
MODO = 'simetrias'

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for m in ('left_margin', 'right_margin'):
    setattr(sec, m, Cm(2.2))
sec.top_margin = sec.bottom_margin = Cm(2.0)
st = doc.styles['Normal']
st.font.name = 'Calibri'
st.font.size = Pt(11)
st.paragraph_format.space_after = Pt(6)
ANCHO = Cm(16.6)


def h(texto, nivel=1):
    doc.add_heading(texto, level=nivel)


def p(texto, negrita_prefijo=None):
    par = doc.add_paragraph()
    if negrita_prefijo:
        par.add_run(negrita_prefijo).bold = True
    par.add_run(texto)
    return par


def bullet(texto, negrita_prefijo=None):
    par = doc.add_paragraph(style='List Bullet')
    if negrita_prefijo:
        par.add_run(negrita_prefijo).bold = True
    par.add_run(texto)


def codigo(texto):
    for linea in texto.strip('\n').split('\n'):
        par = doc.add_paragraph()
        par.paragraph_format.space_after = Pt(0)
        par.paragraph_format.left_indent = Cm(0.5)
        r = par.add_run(linea if linea else ' ')
        r.font.name = 'Consolas'
        r.font.size = Pt(9)
        r._element.rPr.rFonts.set(qn('w:eastAsia'), 'Consolas')
        sombrear(par, 'F2F2F0')
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def sombrear(par, color):
    pPr = par._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), color)
    pPr.append(shd)


def celda_sombra(celda, color):
    tcPr = celda._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), color)
    tcPr.append(shd)


def tabla(encabezado, filas, anchos=None):
    t = doc.add_table(rows=1, cols=len(encabezado))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, txt in enumerate(encabezado):
        c = t.rows[0].cells[i]
        c.text = ''
        c.paragraphs[0].add_run(str(txt)).bold = True
        celda_sombra(c, 'E8EEF7')
    for fila in filas:
        cs = t.add_row().cells
        for i, txt in enumerate(fila):
            cs[i].text = str(txt)
    if anchos:
        for fila in t.rows:
            for i, w in enumerate(anchos):
                fila.cells[i].width = Cm(w)
    for fila in t.rows:
        for c in fila.cells:
            for par in c.paragraphs:
                par.paragraph_format.space_after = Pt(0)
                for r in par.runs:
                    r.font.size = Pt(9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figura(nombre, pie):
    ruta = os.path.join(GRAF, nombre)
    if not os.path.exists(ruta):
        p(f'[Falta la grafica {nombre}: corre evaluacion/graficas.py]')
        return
    doc.add_picture(ruta, width=ANCHO)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(pie)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x52, 0x51, 0x4E)


manif = json.load(open(MANIF, encoding='utf-8')) if os.path.exists(MANIF) else {'args': {}}
A = manif.get('args', {})
res = json.load(open(RES, encoding='utf-8')) if os.path.exists(RES) else {}

# ------------------------------------------------------------------------------- portada
t = doc.add_heading('Red neuronal pura para 5x5 con 4 en raya', 0)
p('Documentación del código, del entrenamiento y de la evaluación contra el Oracle')
p(f'Generado el {datetime.date.today().isoformat()} a partir de los archivos del proyecto.')

# --------------------------------------------------------------------------------- resumen
h('1. Resumen')
p('El proyecto entrena una red neuronal convolucional (CNN residual) para jugar 5x5 con 4 en raya '
  'aprendiendo solo de partidas contra sí misma (self-play) con aprendizaje por refuerzo (PPO). '
  'No se usa ningún tipo de búsqueda: ni MCTS, ni minimax, ni alfa-beta, ni el Oracle, ni un maestro. '
  'Al jugar, la red decide con una sola pasada hacia adelante (forward) y elige la casilla legal con '
  'mayor puntaje. El Oracle (minimax exacto con límite de tiempo) se usa únicamente para medir a la red.')
bullet('modelo final largo1_final: 6466 updates, 64 canales x 5 bloques residuales, 392 126 parámetros.', 'Modelo: ')
bullet('primer checkpoint guardado (largo1_gen500, update 500) y último (largo1_final, update 6466).', 'Checkpoints: ')
bullet('1000 partidas pareadas contra el Oracle con 2, 4 y 10 segundos por jugada, para cada checkpoint.', 'Evaluación: ')

# ------------------------------------------------------------------------------ estructura
h('2. Estructura del proyecto')
p('La carpeta contiene únicamente el código que se usa para entrenar, jugar, evaluar y graficar.')
tabla(['Archivo', 'Para qué sirve'], [
    ['app/game_rules.py', 'Reglas del 5x5 con 4 en raya: jugadas legales, aplicar jugada, ganador.'],
    ['app/az_resnet_model.py', 'La red: CNN residual con cabeza de política y cabeza de valor; codificación del tablero.'],
    ['app/inferencia.py', 'Cómo elige jugada la red: modo puro (1 forward) o con simetrías. Sin búsqueda.'],
    ['app/jugar_az_nosearch_resnet.py', 'Jugar contra la red en la consola.'],
    ['training/train_az_nosearch_v23.py', 'Entrenamiento: self-play + PPO/GAE. Guarda checkpoints y el historial.'],
    ['training/selfplay_vec.py', 'Motor de self-play vectorizado (muchas partidas a la vez) y cálculo de ventajas GAE.'],
    ['training/test_selfplay_vec.py', 'Pruebas del motor de self-play.'],
    ['training/AUDIT_AZ_NOSEARCH_RESNET.py', 'Auditoría: verifica que el entrenamiento y la inferencia no usan búsqueda.'],
    ['evaluacion/oracle_minimax.py', 'El Oracle: minimax exacto con poda alfa-beta, tabla de transposición y límite de tiempo.'],
    ['evaluacion/partidas_vs_oracle.py', 'Juega las 1000 partidas por enfrentamiento en paralelo y guarda resultados.'],
    ['evaluacion/graficas.py', 'Genera todas las gráficas.'],
    ['interfaz/app_gui.py', 'Interfaz gráfica: ver a la red contra el Oracle (2/4/10 s), jugar contra la red y ver resultados.'],
    ['docs/generar_documentacion.py', 'Genera este documento.'],
    ['modelos/', 'largo1_gen500.pt (primer checkpoint) y largo1_final.pt (último).'],
    ['resultados/entrenamiento/', 'Log del entrenamiento, historial de evaluaciones y manifiesto de pureza.'],
    ['resultados/partidas/', 'Resultados de las partidas contra el Oracle (una línea por partida + resumen).'],
    ['resultados/graficas/', 'Gráficas generadas.'],
], anchos=[6.2, 10.4])

# ------------------------------------------------------------------------------ como correr
h('3. Cómo se ejecuta')
p('Todos los comandos se corren desde la carpeta indicada. Requisitos: Python 3.10+, '
  'torch, numpy, matplotlib y python-docx (pip install torch numpy matplotlib python-docx).')
h('3.1 Verificar el código', 2)
codigo('cd training\npython test_selfplay_vec.py\npython AUDIT_AZ_NOSEARCH_RESNET.py')
h('3.2 Entrenamiento (cómo se obtuvo largo1)', 2)
codigo(f'cd training\npython train_az_nosearch_v23.py --preset full --games {A.get("games", 512)} '
       f'--minutes {A.get("minutes", 90):g} --seed {A.get("seed", 1)} --eval-every {A.get("eval_every", 500)} --out largo1')
p('Usa la GPU automáticamente si existe (largo1 se entrenó en GPU con CUDA). Guarda modelos/largo1_genN.pt '
  'cada --eval-every updates y modelos/largo1_final.pt al terminar.')
h('3.3 Partidas contra el Oracle', 2)
codigo('cd evaluacion\npython partidas_vs_oracle.py --procesos 6')
p('Por defecto juega el primer y el último checkpoint contra el Oracle con 2, 4 y 10 s, 1000 partidas cada '
  'enfrentamiento, con la red en modo simetrías. Es reanudable: si se interrumpe, se vuelve a correr el mismo '
  'comando. Conviene usar como máximo un proceso por núcleo libre para no quitarle CPU al Oracle.')
h('3.4 Gráficas y documento', 2)
codigo('cd evaluacion\npython graficas.py\ncd ..\\docs\npython generar_documentacion.py')
h('3.5 Jugar contra la red', 2)
codigo('cd app\npython jugar_az_nosearch_resnet.py --model largo1_final --simetrias')
h('3.6 Interfaz gráfica', 2)
codigo('python interfaz/app_gui.py')
p('Tres pestañas. Red vs Oracle: se elige el tiempo del Oracle (2, 4 o 10 s), el checkpoint, el modo de la red, '
  'quién empieza y las jugadas al azar; se ve en vivo el mapa de calor de la política de la red, su evaluación '
  '(cabeza de valor), su confianza, los nodos y el tiempo del Oracle y si resolvió exacto, gráficas de la partida, '
  'el historial de jugadas y el marcador de la sesión; se pueden jugar series de 10 partidas y guardar partidas. '
  'Tú vs Red: se juega con clics, se puede ver qué jugaría la red y deshacer. Resultados: las 1000 partidas por '
  'enfrentamiento y las curvas del entrenamiento. El Oracle corre en un hilo aparte para que la ventana no se congele.')

# ------------------------------------------------------------------------------ reglas
h('4. El juego (app/game_rules.py)')
p('Tablero de 5x5 (casillas 0 a 24 por filas). X (jugador 1) mueve primero, O es el jugador 2. Gana quien '
  'complete 4 en línea horizontal, vertical o diagonal; hay 28 líneas ganadoras. Si se llena el tablero sin '
  'ganador es empate. Con juego perfecto de ambos lados la partida termina en empate, por lo que contra un rival '
  'perfecto el mejor resultado posible es no perder (salvo que la apertura aleatoria deje una posición ganada).')

# ------------------------------------------------------------------------------- red
h('5. La red neuronal (app/az_resnet_model.py)')
p('Es una red convolucional residual (estilo ResNet / AlphaZero) que recibe el tablero y devuelve dos cosas: '
  'un puntaje por casilla (política) y una estimación de quién va ganando (valor).')
tabla(['Parte', 'Detalle'], [
    ['Entrada', '3 planos de 5x5 desde el punto de vista del jugador que mueve: mis fichas, fichas del rival, casillas vacías.'],
    ['Tronco (stem)', 'Convolución 3x3 de 3 a 64 canales + GroupNorm + ReLU.'],
    ['Bloques residuales', '5 bloques: conv 3x3 -> GroupNorm -> ReLU -> conv 3x3 -> GroupNorm, suma con la entrada (atajo) y ReLU.'],
    ['Cabeza de política', 'Totalmente convolucional: conv 3x3 (64->32) + GroupNorm + ReLU, conv 1x1 (32->1) y un sesgo aprendido por casilla. Da 25 logits.'],
    ['Cabeza de valor', 'conv 1x1 (64->1) + GroupNorm + ReLU, capa densa 25->64 + ReLU, densa 64->1 y tanh. Valor en [-1, 1].'],
    ['Tamaño', '392 126 parámetros entrenables.'],
], anchos=[4.0, 12.6])
p('La cabeza de política convolucional hace que un patrón táctico aprendido en una zona del tablero sirva en '
  'cualquier otra. La cabeza de valor solo se usa durante el entrenamiento (como línea base de PPO); al jugar '
  'no se usa.')

# --------------------------------------------------------------------------- entrenamiento
h('6. Entrenamiento (training/train_az_nosearch_v23.py y selfplay_vec.py)')
h('6.1 De dónde salen los datos: self-play', 2)
bullet(f'en cada update se juegan {A.get("games", 512)} partidas en paralelo; cada jugada de todas las partidas '
       'se calcula con un solo forward en lote.', 'Partidas en paralelo: ')
bullet(f'cada partida empieza con 0 a {A.get("max_open", 4)} jugadas al azar para ver posiciones variadas.', 'Aperturas aleatorias: ')
bullet('se aprende de las jugadas de X y de O.', 'Ambos lados: ')
bullet('cada decisión se toma viendo el tablero en una rotación o reflejo al azar (8 simetrías). Es aumento '
       'de datos y sigue siendo exactamente on-policy.', 'Simetrías: ')
bullet(f'el {100 * A.get("pool_prob", 0.25):.0f} % de las partidas se juega contra copias anteriores de la propia red '
       f'(una copia cada {A.get("pool_every", 25)} updates, se guardan las últimas {A.get("pool_size", 12)}), para no '
       'olvidar cómo ganarle a estilos pasados.', 'Liga de versiones anteriores: ')
bullet('las jugadas se muestrean de la política (exploración); al jugar de verdad se usa la mejor.', 'Muestreo: ')
h('6.2 Recompensa y ventajas (GAE)', 2)
p('La única recompensa es el resultado final: +1 si gana, -1 si pierde, 0 si empata, asignada a la última '
  'decisión de cada lado. Las ventajas se calculan con GAE (Generalized Advantage Estimation):')
codigo('delta_t = gamma * V(s_t+1) - V(s_t)        (en la última jugada: r - V(s_t))\n'
       'A_t     = delta_t + gamma * lambda * A_t+1\n'
       f'gamma = {A.get("gamma", 0.99)},  lambda = {A.get("gae_lambda", 0.95)}')
h('6.3 Función de pérdida (PPO)', 2)
codigo('ratio_t   = pi_nueva(a_t|s_t) / pi_vieja(a_t|s_t)\n'
       'L_pol     = -media( min(ratio_t * A_t,  clip(ratio_t, 1-eps, 1+eps) * A_t) )\n'
       'L_valor   = 0.5 * media( max((V - R)^2, (V_recortado - R)^2) )\n'
       'H         = entropía de la política (sobre casillas legales)\n'
       'L_total   = L_pol + c_v * L_valor - c_ent * H')
p(f'eps = {A.get("ppo_clip", 0.2)}, c_v = {A.get("value_coef", 0.5)}, c_ent baja linealmente de '
  f'{A.get("entropy_start", 0.03)} a {A.get("entropy_end", 0.003)}. Las ventajas se normalizan por lote, los '
  f'gradientes se recortan a norma 1 y la actualización se detiene antes si la KL supera '
  f'1.5 x {A.get("target_kl", 0.015)}.')
h('6.4 Hiperparámetros de largo1', 2)
tabla(['Parámetro', 'Valor', 'Parámetro', 'Valor'], [
    ['Preset', A.get('preset', 'full'), 'Canales x bloques', f'{A.get("channels", 64)} x {A.get("blocks", 5)}'],
    ['Partidas por update', A.get('games', 512), 'Épocas PPO', A.get('ppo_epochs', 4)],
    ['Minibatch', A.get('minibatch', 1024), 'Optimizador', 'AdamW (weight decay 1e-4)'],
    ['Learning rate', f'{A.get("lr", 1.5e-4)} (coseno hasta x{A.get("lr_final_frac", 0.05)})', 'Recorte PPO', A.get('ppo_clip', 0.2)],
    ['Target KL', A.get('target_kl', 0.015), 'Recorte de valor', A.get('vf_clip', 0.5)],
    ['Entropía', f'{A.get("entropy_start", 0.03)} -> {A.get("entropy_end", 0.003)}', 'Aperturas aleatorias', f'0..{A.get("max_open", 4)}'],
    ['Presupuesto', f'{A.get("minutes", 90):g} min (GPU)', 'Updates realizados', '6466'],
    ['Semilla', A.get('seed', 1), 'Checkpoint cada', f'{A.get("eval_every", 500)} updates'],
], anchos=[3.8, 4.5, 3.8, 4.5])

h('6.5 Cómo fue el entrenamiento', 2)
figura('01_funcion_de_perdida.png', 'Figura 1. Función de pérdida total de PPO y sus tres términos a lo largo del entrenamiento.')
bullet('baja de ~0,34 a ~0,02: la red aprende a predecir el resultado de la partida.', 'Pérdida de valor: ')
bullet('oscila alrededor de 0, como es normal en PPO (las ventajas se normalizan); no es una medida de '
       'calidad por sí sola.', 'Pérdida de política: ')
bullet('cae al inicio (la red deja de jugar al azar) y luego sube un poco y se estabiliza: contra rivales '
       'fuertes muchas jugadas son igual de buenas.', 'Entropía: ')
figura('02_progreso_entrenamiento.png', 'Figura 2. Empates en self-play, KL, fracción de muestras recortadas y learning rate.')
bullet('pasan de 0 % a ~60-65 % hacia el update 1000 y se mantienen: como el juego es empate con juego '
       'perfecto, que la red empate cada vez más contra sí misma indica que juega cada vez mejor.', 'Empates en self-play: ')
bullet('la KL y la fracción recortada bajan al final porque el learning rate decae (coseno).', 'Estabilidad: ')

# --------------------------------------------------------------------------- inferencia
h('7. Cómo juega la red (app/inferencia.py)')
bullet('el tablero entra a la red una vez y se elige la casilla legal con mayor puntaje.', 'Modo puro: ')
bullet('la misma red recibe en un solo forward (lote de 8) el tablero en sus 8 rotaciones y reflejos; cada '
       'salida se devuelve a la orientación real y se promedian las log-probabilidades. No se simula ninguna '
       'jugada ni se mira el futuro: es la misma red vista desde 8 ángulos, la misma simetría que se usa como '
       'aumento de datos al entrenar.', 'Modo simetrías: ')
p('En las partidas contra el Oracle se usa el modo simetrías porque comete menos errores. En pruebas previas '
  'contra un solver exacto (200 partidas), con largo1_final las derrotas evitables bajaron de 23 a 18.')

# --------------------------------------------------------------------------- evaluacion
h('8. Evaluación contra el Oracle (evaluacion/)')
h('8.1 El Oracle', 2)
p('oracle_minimax.py es un minimax exacto con poda alfa-beta, tabla de transposición con clave canónica '
  '(las 8 simetrías comparten entrada) y orden de jugadas ganar > bloquear > centro. Siempre busca hasta el '
  'final de la partida. Si se le acaba el tiempo, juega la mejor jugada entre las que alcanzó a resolver; por '
  'eso se reporta qué porcentaje de sus jugadas fueron exactas.')
h('8.2 Diseño de las partidas', 2)
bullet('cada apertura aleatoria (1, 2 o 3 jugadas) se juega dos veces, una con la red como X y otra como O.', 'Pareadas: ')
bullet('las aperturas dependen solo de la semilla y del número de par: ambos checkpoints y los tres tiempos '
       'juegan exactamente las mismas aperturas.', 'Mismas aperturas: ')
bullet('1000 partidas (500 pares) por enfrentamiento; intervalos de confianza del 95 % de Wilson.', 'Tamaño: ')
bullet('se reparten en varios procesos, cada uno con su propio Oracle.', 'En paralelo: ')

# ----------------------------------------------------------------------------- resultados
h('9. Resultados contra el Oracle')
claves = [(m, t) for m in MODELOS for t in TIEMPOS if f'{m}__{MODO}__t{t:g}s' in res]
if claves:
    filas = []
    for m, t in claves:
        r = res[f'{m}__{MODO}__t{t:g}s']
        filas.append(['Primer (gen500)' if m == MODELOS[0] else 'Último (final)', f'{t:g} s', r['partidas'],
                      f'{r["victorias"]} ({r["pct_victorias"]:.1f} %)', f'{r["empates"]} ({r["pct_empates"]:.1f} %)',
                      f'{r["derrotas"]} ({r["pct_derrotas"]:.1f} %)',
                      f'{r["ic95_derrotas"][0]:.1f}-{r["ic95_derrotas"][1]:.1f} %',
                      f'{r["pct_jugadas_oracle_exactas"]:.0f} %' if r['pct_jugadas_oracle_exactas'] is not None else '-'])
    tabla(['Checkpoint', 'Oracle', 'Partidas', 'Victorias', 'Empates', 'Derrotas', 'IC95 derrotas', 'Oracle exacto'],
          filas, anchos=[2.6, 1.4, 1.5, 2.3, 2.3, 2.3, 2.3, 1.9])
    h('Conclusiones de las partidas', 2)
    for t in TIEMPOS:
        k0, k1 = f'{MODELOS[0]}__{MODO}__t{t:g}s', f'{MODELOS[-1]}__{MODO}__t{t:g}s'
        if k0 in res and k1 in res:
            a, b = res[k0], res[k1]
            sep = ('los intervalos de confianza no se cruzan, así que la mejora es estadísticamente clara'
                   if b['ic95_derrotas'][1] < a['ic95_derrotas'][0] else 'los intervalos de confianza se cruzan')
            bullet(f'las derrotas bajan de {a["pct_derrotas"]:.1f} % (primer checkpoint) a {b["pct_derrotas"]:.1f} % '
                   f'(último), y las victorias pasan de {a["pct_victorias"]:.1f} % a {b["pct_victorias"]:.1f} %; {sep}.',
                   f'Oracle {t:g} s: ')
    ult = [res[f'{MODELOS[-1]}__{MODO}__t{t:g}s'] for t in TIEMPOS if f'{MODELOS[-1]}__{MODO}__t{t:g}s' in res]
    if len(ult) > 1:
        bullet(f'con más tiempo el Oracle resuelve exacto más jugadas ({ult[0]["pct_jugadas_oracle_exactas"]:.0f} % -> '
               f'{ult[-1]["pct_jugadas_oracle_exactas"]:.0f} %) y por eso la red gana un poco menos y pierde un poco más '
               f'({ult[0]["pct_derrotas"]:.1f} % -> {ult[-1]["pct_derrotas"]:.1f} % de derrotas).', 'Efecto del tiempo: ')
    figura('03_resumen_vs_oracle.png', 'Figura 3. Victorias, empates y derrotas de cada enfrentamiento.')
    for t in TIEMPOS:
        figura(f'04_enfrentamiento_t{t:g}s.png', f'Primer vs último checkpoint contra el Oracle con {t:g} s por jugada.')
    figura('05_derrotas_ic95.png', 'Porcentaje de derrotas con intervalo de confianza del 95 %.')
    figura('06_oracle_exacto.png', 'Porcentaje de jugadas que el Oracle resolvió de forma exacta en cada enfrentamiento.')
    p('Cómo leer estos números: cuando el Oracle no alcanza a resolver una jugada dentro de su tiempo, juega con '
      'información incompleta y puede equivocarse; las victorias de la red vienen de esos errores. Contra un rival '
      'que juega perfecto no se puede ganar desde una posición de empate, así que la medida más importante es el '
      'porcentaje de derrotas y cómo baja del primer al último checkpoint.')
else:
    p('Todavía no hay resultados de partidas. Corre evaluacion/partidas_vs_oracle.py, luego '
      'evaluacion/graficas.py y vuelve a generar este documento.')

# ------------------------------------------------------------------------------ pureza
h('10. Pureza: por qué la red no hace "trampa"')
bullet('el entrenamiento solo usa la red actual y sus copias anteriores; nunca el Oracle ni otra búsqueda.', 'Entrenamiento: ')
bullet('training/AUDIT_AZ_NOSEARCH_RESNET.py revisa el código de entrenamiento e inferencia y falla si '
       'encuentra imports o llamadas a MCTS, minimax, alfa-beta, Oracle, búsqueda o maestro.', 'Auditoría: ')
bullet('cada checkpoint guarda un manifiesto (mcts_used=False, oracle_used_for_training=False, ...).', 'Manifiesto: ')
bullet('el Oracle vive solo en evaluacion/ y solo se usa para medir.', 'Separación: ')

out = os.path.join(HERE, 'DOCUMENTACION.docx')
doc.save(out)
print('Documento generado:', out)
