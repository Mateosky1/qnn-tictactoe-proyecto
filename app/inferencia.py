"""Como elige su jugada la red (sin busqueda de ningun tipo).

Dos modos, ambos son SOLO la red neuronal:

- pura (simetrias=False): 1 forward del tablero + argmax entre las casillas legales.
- simetrias (simetrias=True): la MISMA red recibe en un unico forward (batch de 8) el
  tablero en sus 8 rotaciones/reflejos. Cada salida se devuelve a la orientacion real y
  se promedian las log-probabilidades; luego argmax entre las casillas legales.
  No se simula ninguna jugada ni se mira el futuro: es la misma red vista desde 8 angulos
  (la misma simetria que ya se usa como aumento de datos durante el entrenamiento).
"""
import numpy as np
import torch

from az_resnet_model import encode
from game_rules import CELLS, BOARD_N, legal_moves

_IDX = np.arange(CELLS).reshape(BOARD_N, BOARD_N)
# SIMETRIAS[k][i] = casilla del tablero real que la vista k pone en la posicion i
SIMETRIAS = [np.ascontiguousarray(f(np.rot90(_IDX, k))).reshape(-1)
             for k in range(4) for f in (lambda a: a, np.fliplr)]


@torch.no_grad()
def jugada_red(model, board, player, simetrias=False):
    """Devuelve la casilla (0..24) que elige la red para `player` en `board`."""
    legal = legal_moves(board)
    if not legal:
        return None
    model.eval()
    x = encode(board, player)                                   # (3, 5, 5)
    if not simetrias:
        logits, _ = model(torch.from_numpy(x[None]))
        return int(legal[int(torch.argmax(logits[0, legal]).item())])
    xf = x.reshape(3, CELLS)
    vistas = np.stack([xf[:, s] for s in SIMETRIAS]).reshape(8, 3, BOARD_N, BOARD_N)
    logp = torch.log_softmax(model(torch.from_numpy(np.ascontiguousarray(vistas)))[0], 1).numpy()
    puntaje = np.zeros(CELLS)
    for s, lp in zip(SIMETRIAS, logp):
        puntaje[s] += lp                                        # vuelve a la orientacion real
    return int(legal[int(np.argmax(puntaje[legal]))])


@torch.no_grad()
def analizar(model, board, player, simetrias=False):
    """Lo mismo que jugada_red, pero devuelve tambien lo que 'piensa' la red (para la interfaz).

    probs : np.array(25) probabilidad de la politica en cada casilla legal (0 en ocupadas)
    valor : estimacion de la cabeza de valor en [-1, 1] para `player` (+1 = cree que gana)
    jugada: la casilla que elige (identica a jugada_red)
    """
    legal = legal_moves(board)
    model.eval()
    x = encode(board, player)
    if not simetrias:
        logits, v = model(torch.from_numpy(x[None]))
        puntaje = logits[0].numpy().astype(np.float64)
        valor = float(v[0])
    else:
        xf = x.reshape(3, CELLS)
        vistas = np.stack([xf[:, s] for s in SIMETRIAS]).reshape(8, 3, BOARD_N, BOARD_N)
        logits, v = model(torch.from_numpy(np.ascontiguousarray(vistas)))
        logp = torch.log_softmax(logits, 1).numpy()
        puntaje = np.zeros(CELLS)
        for s, lp in zip(SIMETRIAS, logp):
            puntaje[s] += lp
        valor = float(v.mean())
    probs = np.zeros(CELLS)
    if legal:
        z = puntaje[legal] - puntaje[legal].max()
        e = np.exp(z if not simetrias else z / len(SIMETRIAS))
        probs[legal] = e / e.sum()
    jugada = int(legal[int(np.argmax(puntaje[legal]))]) if legal else None
    return {'probs': probs, 'valor': valor, 'jugada': jugada}
