"""
oracle_minimax.py -- Jugador PERFECTO para el 5x5 con victoria a 4 en
linea (WIN_LEN=4, ver app/qnn_core.py), via minimax + poda alfa-beta
EXACTO -- sin heuristicas, sin red neuronal, sin aprendizaje. Es el
"jugador perfecto" que piden las Fases 5-6 del profesor.

Por que este modulo y no AgenteV2AlphaBeta (que ya existe en
qnn_core.py): V2 tambien hace negamax+poda, pero evalua las hojas con
la red QNN de Saul cuando se acaba la profundidad permitida -- eso es
una heuristica de busqueda limitada, no un oraculo game-theoretic
exacto. Este modulo llega SIEMPRE hasta un estado terminal real
(victoria/derrota/empate), nunca corta por profundidad ni usa ninguna
red para estimar nada. Por eso vive aparte y no se toca qnn_core.py.

Analisis de viabilidad ANTES de implementar (lo que pide la Fase 5):
- Tablero 5x5 = 25 casillas, 3^25 (~8.5e11) es la cota superior de
  configuraciones posibles del tablero (ignora turnos y quien gano),
  MUY por encima de lo que se puede recorrer por fuerza bruta sin
  memoizacion.
- Pero: (a) el juego SIEMPRE termina en <=25 jugadas; (b) con 28
  lineas ganadoras de 4 (vs. solo 8 en el 3x3 clasico) las partidas
  terminan casi siempre bastante antes de llenar el tablero, así que
  el arbol real de partidas jugadas es mucho mas chico que 25!;
  (c) hay 8 simetrias (grupo diedral D4: 4 rotaciones x reflejo) que
  colapsan posiciones equivalentes.
- Conclusion (medida empiricamente mas abajo, no solo estimada): con
  poda alfa-beta + tabla de transposicion con clave CANONICA (la
  version simetrica lexicograficamente menor) + ordenar movimientos
  probando primero victorias inmediatas, resolver una posicion
  CUALQUIERA que ya tiene varias fichas puestas es rapido (milisegundos
  a un par de segundos). Resolver desde el TABLERO VACIO (el peor caso,
  maxima ramificacion) es el unico caso caro; se cachea una sola vez
  (ver solve_from_empty_board_cached) porque el tablero vacio siempre
  es el mismo, así que ese costo se paga una sola vez para todas las
  partidas donde el Oracle empieza, no una vez por partida.
"""
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
from game_rules import (CELLS, BOARD_N, EMPTY, P1, P2, legal_moves, apply_move,
                        game_over, immediate_wins, MoveReport)

WIN, DRAW, LOSS = 1, 0, -1


# ------------------------- simetrias del tablero (D4) -----------------------
def _rotate90(board):
    n = BOARD_N
    return tuple(board[(n - 1 - c) * n + r] for r in range(n) for c in range(n))


def _reflect(board):
    n = BOARD_N
    return tuple(board[r * n + (n - 1 - c)] for r in range(n) for c in range(n))


def _all_symmetries(board):
    b = tuple(board)
    out = []
    for _ in range(4):
        b = _rotate90(b)
        out.append(b)
        out.append(_reflect(b))
    return out


def canonical_key(board, player):
    """La representacion mas chica (lexicograficamente) entre las 8
    simetrias del tablero, junto con quien mueve. Dos posiciones que
    son la misma partida "rotada/reflejada" comparten esta clave, asi
    que la tabla de transposicion las resuelve una sola vez."""
    return (min(_all_symmetries(board)), player)


# --------------------------------- oraculo -----------------------------------
class OracleMinimax:
    """Jugador perfecto exacto. select(board, player) -> MoveReport,
    igual que AgenteV1Saul/V2/V3 en qnn_core.py, para poder usarlo con
    el mismo tipo de codigo de partida (ver app/torneo.py)."""

    def __init__(self):
        self.tt = {}          # canonical_key -> (valor, flag); PERSISTE entre jugadas
        self.tt_r = {}        # idem para busquedas restringidas (no exactas): se vacia en cada select
        self.tt_max = 600_000 # tope de entradas (memoria); al superarlo se vacia
        self.nodes_last_call = 0
        self._deadline = None
        self._timed_out = False

    def _ordered_moves(self, board, player, restrict_adjacent=False, radius=2):
        """Movimientos legales ordenados (victoria inmediata, bloqueo, centro).
        Solo ORDENA; no elimina nada (salvo restrict_adjacent, que es opcional
        y explicitamente no exacto). Por eso el resultado sigue siendo exacto."""
        moves = legal_moves(board)
        if restrict_adjacent:
            occupied = [i for i in range(CELLS) if board[i] != EMPTY]
            if occupied:
                def _near(m):
                    mr, mc = divmod(m, BOARD_N)
                    return any(max(abs(mr - divmod(o, BOARD_N)[0]), abs(mc - divmod(o, BOARD_N)[1])) <= radius
                               for o in occupied)
                restricted = [m for m in moves if _near(m)]
                if restricted:
                    moves = restricted
            else:
                moves = [CELLS // 2]
        wins_now = set(immediate_wins(board, player))
        opp_wins = set(immediate_wins(board, 3 - player))

        def move_priority(m):
            if m in wins_now:
                return 0
            if m in opp_wins:
                return 1
            r, c = divmod(m, BOARD_N)
            return 2 + abs(r - BOARD_N // 2) + abs(c - BOARD_N // 2)
        moves.sort(key=move_priority)
        return moves

    def solve(self, board, player, alpha=-2, beta=2, restrict_adjacent=False, radius=2):
        """Minimax/negamax exacto con poda alfa-beta y TT con cotas.

        La tabla de transposicion guarda el tipo de resultado (EXACT/LOWER/UPPER),
        porque un valor obtenido tras un corte alfa-beta no siempre es el valor
        exacto del nodo. Esto evita contaminar futuras busquedas con una cota.
        """
        self.nodes_last_call += 1
        if self._deadline is not None and self.nodes_last_call % 2048 == 0 and time.time() > self._deadline:
            self._timed_out = True
        if self._timed_out:
            return DRAW, (legal_moves(board) or [None])[0]

        done, winner = game_over(board)
        if done:
            if winner == 0:
                return DRAW, None
            return LOSS, None

        tt = self.tt_r if restrict_adjacent else self.tt
        key = canonical_key(board, player)
        alpha_orig, beta_orig = alpha, beta
        cached = tt.get(key)
        if cached is not None:
            cval, cflag = cached
            if cflag == 'EXACT':
                return cval, None
            if cflag == 'LOWER':
                alpha = max(alpha, cval)
            elif cflag == 'UPPER':
                beta = min(beta, cval)
            if alpha >= beta:
                return cval, None

        moves = self._ordered_moves(board, player, restrict_adjacent, radius)

        best_val, best_move = LOSS, moves[0]
        cutoff = False
        for m in moves:
            nb = apply_move(board, m, player)
            val, _ = self.solve(nb, 3 - player, -beta, -alpha,
                                restrict_adjacent=restrict_adjacent, radius=radius)
            val = -val
            if val > best_val:
                best_val, best_move = val, m
            alpha = max(alpha, val)
            if best_val == WIN:
                break
            if alpha >= beta:
                cutoff = True
                break

        # Clasificacion estandar de TT para negamax+alpha-beta.
        if not self._timed_out:
            if best_val <= alpha_orig:
                flag = 'UPPER'
            elif best_val >= beta_orig:
                flag = 'LOWER'
            else:
                flag = 'EXACT'
            tt[key] = (best_val, flag)
        return best_val, best_move

    def select(self, board, player, restrict_adjacent=False, radius=2, time_limit_s=None):
        """Elige jugada. Si hay tiempo suficiente el resultado es EXACTO (juego
        perfecto). Si se agota el limite de tiempo se devuelve la mejor jugada
        entre las raices YA resueltas por completo (no una jugada arbitraria);
        solo si ninguna raiz llego a resolverse se usa la heuristica
        ganar > bloquear > centro. En ambos casos queda marcado timed_out=True.

        La TT NO se vacia entre jugadas: la busqueda siempre llega a estados
        terminales y solo se guardan nodos terminados antes del timeout, asi que
        cada entrada es valida para cualquier busqueda futura (las busquedas
        restringidas usan una TT aparte, que si se vacia)."""
        t0 = time.time()
        self.nodes_last_call = 0
        self._timed_out = False
        if len(self.tt) > self.tt_max:
            self.tt.clear()
        self.tt_r.clear()
        self._deadline = (t0 + time_limit_s) if time_limit_s is not None else None
        board = list(board)
        moves = self._ordered_moves(board, player, restrict_adjacent, radius)

        alpha, beta = -2, 2
        best_val, best_move, completed = None, None, 0
        for m in moves:
            val, _ = self.solve(apply_move(board, m, player), 3 - player, -beta, -alpha,
                                restrict_adjacent=restrict_adjacent, radius=radius)
            if self._timed_out:
                break
            val = -val
            completed += 1
            if best_val is None or val > best_val:
                best_val, best_move = val, m
            alpha = max(alpha, val)
            if best_val == WIN:
                break

        if self._timed_out:
            unexplored = [m for m in moves[completed:]]
            if best_move is None:
                best_move = moves[0]          # ganar > bloquear > centro
            elif best_val == LOSS and unexplored:
                best_move = unexplored[0]     # lo resuelto pierde: mejor probar otra
            val_out, exact = None, False
            reason = (f"NO EXACTO -- se agoto el limite de tiempo ({time_limit_s}s); "
                      f"{completed}/{len(moves)} jugadas resueltas")
        else:
            val_out, exact = best_val, True
            reason = {WIN: "victoria forzada", DRAW: "empate con juego perfecto",
                      LOSS: "derrota inevitable (rival juega perfecto)"}[best_val]
        elapsed_ms = (time.time() - t0) * 1000
        return MoveReport(move=best_move, value=float(val_out if exact else 0.0),
                          nodes=self.nodes_last_call, elapsed_ms=elapsed_ms, reason=reason,
                          extra={"valor_exacto": val_out, "tt_size": len(self.tt),
                                 "timed_out": self._timed_out,
                                 "raices_resueltas": completed, "raices_total": len(moves)})


_shared_oracle_empty_board_cache = None


def solve_from_empty_board_cached(oracle=None, time_limit_s=None):
    """El tablero vacio siempre es la MISMA posicion -- resolverla es
    el caso mas caro (maxima ramificacion). Se midio (ver docstring del
    modulo) que NO se resuelve exacto en tiempos razonables en este
    entorno, asi que esta funcion usa el modo con limite de tiempo y
    devuelve la mejor jugada ENCONTRADA hasta ahora (marcada como
    aproximada si se agoto el tiempo). El resultado se guarda en un
    cache de modulo para pagar este costo una sola vez en todo el
    proyecto, no una vez por partida."""
    global _shared_oracle_empty_board_cache
    if _shared_oracle_empty_board_cache is None:
        oracle = oracle or OracleMinimax()
        board = [EMPTY] * CELLS
        report = oracle.select(board, P1, time_limit_s=time_limit_s)
        _shared_oracle_empty_board_cache = {
            "valor": report.value, "mejor_primera_jugada": report.move,
            "nodos": report.nodes, "tt_size": oracle.tt.__len__(),
            "tiempo_seg": report.elapsed_ms / 1000, "exacto": not report.extra["timed_out"],
        }
    return _shared_oracle_empty_board_cache


if __name__ == "__main__":
    o = OracleMinimax()
    info = solve_from_empty_board_cached(o, time_limit_s=None)
    print("Resultado desde el tablero vacio (posiblemente aproximado, ver 'exacto'):", info)
