"""Partidas de la red contra el Oracle (minimax exacto con limite de tiempo).

El Oracle SOLO se usa aqui, para evaluar. El entrenamiento nunca lo toca.

Diseno de la evaluacion
- Partidas PAREADAS: cada apertura aleatoria (1, 2 o 3 jugadas al azar) se juega dos veces,
  una con la red como X y otra con la red como O. Asi ningun bando tiene ventaja.
- Las aperturas dependen solo de --seed-base y del numero de par: el primer y el ultimo
  checkpoint, y los tres limites de tiempo, juegan EXACTAMENTE las mismas aperturas.
- Se reparte en varios procesos (--procesos). Cada proceso tiene su propio Oracle.
  Usa como maximo un proceso por nucleo libre: si los procesos compiten por CPU, el Oracle
  piensa menos dentro de su limite de tiempo y la medicion sale sesgada.
- Reanudable: cada partida se guarda al terminar (archivo .jsonl). Si se corta, vuelve a
  correr el mismo comando y continua donde quedo.

Uso (desde evaluacion/):
    python partidas_vs_oracle.py                         # todo: 2 checkpoints x 2/4/10 s x 1000 partidas
    python partidas_vs_oracle.py --procesos 6
    python partidas_vs_oracle.py --modelos largo1_final --tiempos 2 --partidas 100   # prueba corta

Salida: ../resultados/partidas/<modelo>__<modo>__t<seg>s.jsonl  (una linea por partida)
        ../resultados/partidas/resumen.json                      (totales por enfrentamiento)
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'app'))
sys.path.insert(0, HERE)

from game_rules import EMPTY, P1, P2, CELLS, legal_moves, apply_move, game_over  # noqa: E402

APERTURAS = (1, 2, 3)


# ----------------------------------------------------------------------------- aperturas
def apertura(seed_base, par):
    """Tablero con 1-3 jugadas al azar. Depende solo de (seed_base, par)."""
    rng = random.Random(f'open-{seed_base}-{par}')
    n = rng.choice(APERTURAS)
    board, p = [EMPTY] * CELLS, P1
    for _ in range(n):
        board = apply_move(board, rng.choice(legal_moves(board)), p)
        p = 3 - p
    return n, board, p


# ------------------------------------------------------------------- trabajo de cada proceso
_W = {}


def _init_worker(ruta_modelo, simetrias, tiempo):
    import torch
    torch.set_num_threads(1)
    from az_resnet_model import load_checkpoint
    from oracle_minimax import OracleMinimax
    _W['model'], _ = load_checkpoint(ruta_modelo)
    _W['oracle'] = OracleMinimax()
    _W['simetrias'], _W['tiempo'] = simetrias, tiempo


def _jugar(board0, to_move, red_es_x):
    from inferencia import jugada_red
    red = P1 if red_es_x else P2
    board, p = list(board0), to_move
    jug_oracle = timeouts = plies = 0
    t_oracle = 0.0
    while True:
        done, winner = game_over(board)
        if done:
            res = 'E' if winner == 0 else ('V' if winner == red else 'D')
            return res, jug_oracle, timeouts, plies, t_oracle
        if p == red:
            mv = jugada_red(_W['model'], board, p, _W['simetrias'])
        else:
            rep = _W['oracle'].select(board, p, time_limit_s=_W['tiempo'])
            mv = rep.move
            jug_oracle += 1
            timeouts += int(rep.extra['timed_out'])
            t_oracle += rep.elapsed_ms / 1000
        board = apply_move(board, mv, p)
        p = 3 - p
        plies += 1


def _jugar_par(args):
    seed_base, par = args
    n_open, board, to_move = apertura(seed_base, par)
    orden = [True, False]
    if random.Random(f'order-{seed_base}-{par}').random() < 0.5:
        orden.reverse()
    out = []
    for red_es_x in orden:
        res, jo, to, pl, tor = _jugar(board, to_move, red_es_x)
        out.append({'par': par, 'red_es_x': red_es_x, 'apertura': n_open, 'resultado': res,
                    'jugadas_oracle': jo, 'oracle_sin_tiempo': to, 'plies': pl,
                    'seg_oracle': round(tor, 3)})
    return out


# ----------------------------------------------------------------------------- utilidades
def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def resumir(filas):
    n = len(filas)
    v = sum(f['resultado'] == 'V' for f in filas)
    e = sum(f['resultado'] == 'E' for f in filas)
    d = sum(f['resultado'] == 'D' for f in filas)
    jo = sum(f['jugadas_oracle'] for f in filas)
    to = sum(f['oracle_sin_tiempo'] for f in filas)
    pct = lambda k: round(100 * k / n, 2) if n else 0.0
    ic = lambda k: [round(100 * x, 2) for x in wilson(k, n)]
    return {'partidas': n,
            'victorias': v, 'empates': e, 'derrotas': d,
            'pct_victorias': pct(v), 'pct_empates': pct(e), 'pct_derrotas': pct(d),
            'ic95_victorias': ic(v), 'ic95_derrotas': ic(d), 'pct_no_derrota': pct(v + e),
            'jugadas_oracle': jo,
            'pct_jugadas_oracle_exactas': round(100 * (jo - to) / jo, 2) if jo else None}


def leer_jsonl(ruta):
    if not os.path.exists(ruta):
        return []
    with open(ruta, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def ruta_modelo(nombre):
    return nombre if nombre.endswith('.pt') else os.path.join(ROOT, 'modelos', nombre + '.pt')


# ----------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--modelos', nargs='+', default=['largo1_gen500', 'largo1_final'],
                    help='checkpoints en modelos/ (por defecto el primero y el ultimo)')
    ap.add_argument('--tiempos', nargs='+', type=float, default=[2, 4, 10], help='segundos del Oracle por jugada')
    ap.add_argument('--partidas', type=int, default=1000, help='partidas por enfrentamiento (numero par)')
    ap.add_argument('--modo', choices=['simetrias', 'pura'], default='simetrias',
                    help="'simetrias' = la red promedia las 8 simetrias (por defecto); 'pura' = 1 forward")
    ap.add_argument('--procesos', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--seed-base', type=int, default=50000)
    ap.add_argument('--salida', default=os.path.join(ROOT, 'resultados', 'partidas'))
    a = ap.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    pares = a.partidas // 2
    resumen_path = os.path.join(a.salida, 'resumen.json')
    resumen = json.load(open(resumen_path, encoding='utf-8')) if os.path.exists(resumen_path) else {}
    print(f'{len(a.modelos)} modelo(s) x {len(a.tiempos)} tiempo(s) x {2 * pares} partidas | modo={a.modo} '
          f'| procesos={a.procesos} | seed-base={a.seed_base}', flush=True)

    for modelo in a.modelos:
        for t in a.tiempos:
            clave = f'{modelo}__{a.modo}__t{t:g}s'
            ruta = os.path.join(a.salida, clave + '.jsonl')
            hechas = leer_jsonl(ruta)
            ya = {f['par'] for f in hechas}
            pendientes = [(a.seed_base, i) for i in range(pares) if i not in ya]
            print(f'\n== {clave}: {len(ya)}/{pares} pares hechos, faltan {len(pendientes)}', flush=True)
            t0 = time.time()
            if pendientes:
                with mp.Pool(a.procesos, initializer=_init_worker,
                             initargs=(ruta_modelo(modelo), a.modo == 'simetrias', t)) as pool, \
                        open(ruta, 'a', encoding='utf-8') as fo:
                    for k, filas in enumerate(pool.imap_unordered(_jugar_par, pendientes), 1):
                        for fila in filas:
                            fo.write(json.dumps(fila) + '\n')
                        fo.flush()
                        hechas += filas
                        if k % 25 == 0 or k == len(pendientes):
                            r = resumir(hechas)
                            eta = (time.time() - t0) / k * (len(pendientes) - k)
                            print(f'  [{r["partidas"]}/{2 * pares}] V={r["victorias"]} E={r["empates"]} '
                                  f'D={r["derrotas"]}  faltan ~{eta / 60:.0f} min', flush=True)
            r = resumir(hechas)
            r.update({'modelo': modelo, 'modo': a.modo, 'oracle_seg': t, 'seed_base': a.seed_base})
            resumen[clave] = r
            with open(resumen_path, 'w', encoding='utf-8') as f:
                json.dump(resumen, f, indent=2, ensure_ascii=False)
            print(f'  TOTAL {clave}: V/E/D = {r["victorias"]}/{r["empates"]}/{r["derrotas"]} '
                  f'(derrotas {r["pct_derrotas"]}% IC95 {r["ic95_derrotas"]}) '
                  f'| Oracle exacto en {r["pct_jugadas_oracle_exactas"]}% de sus jugadas', flush=True)
    print(f'\nResumen guardado en {resumen_path}\nSiguiente paso: python graficas.py', flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
