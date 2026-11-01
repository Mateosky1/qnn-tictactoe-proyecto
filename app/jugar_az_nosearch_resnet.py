"""Humano vs Red Neuronal Pura.

La IA usa estrictamente:
tablero -> un forward CNN -> mascara de movimientos legales -> argmax.

No usa MCTS, Minimax, Alpha-Beta, Negamax, Oracle ni busqueda.
"""

import argparse
import os
import random
import sys
import time

import torch

from game_rules import P1, P2, EMPTY, legal_moves, apply_move, game_over
from az_resnet_model import load_checkpoint
from inferencia import jugada_red


def pure_move(model, board, player, simetrias=False):
    """Jugada de la red: ver app/inferencia.py (sin busqueda)."""
    return jugada_red(model, board, player, simetrias)


def show(board):
    """Muestra un tablero 5x5 limpio, sin numeros dentro de las casillas."""
    print()
    print("                 TABLERO 5x5")
    print()
    print("             1   2   3   4   5")
    print("           +---+---+---+---+---+")

    for r in range(5):
        cells = []
        for c in range(5):
            value = board[r * 5 + c]
            if value == P1:
                cells.append(" X ")
            elif value == P2:
                cells.append(" O ")
            else:
                cells.append("   ")

        print(f"        {r + 1}  |" + "|".join(cells) + "|")
        print("           +---+---+---+---+---+")

    print()


def ask_human_move(board):
    """Solicita fila y columna y las convierte a la posicion interna 0..24."""
    legal = legal_moves(board)

    while True:
        try:
            row = int(input("Fila (1-5): "))
            col = int(input("Columna (1-5): "))
        except ValueError:
            print("Entrada no valida. Usa numeros del 1 al 5.")
            print()
            continue

        if not (1 <= row <= 5 and 1 <= col <= 5):
            print("Fila y columna deben estar entre 1 y 5.")
            print()
            continue

        move = (row - 1) * 5 + (col - 1)

        if move not in legal:
            print("Esa casilla ya esta ocupada. Elige otra.")
            print()
            continue

        return move


def ask_who_starts():
    """Menu interactivo: devuelve 'humano', 'ia' o 'azar'."""
    print()
    print("Quien empieza?")
    print("  1) Yo (juego con X)")
    print("  2) La red (yo juego con O)")
    print("  3) Al azar")
    while True:
        choice = input("Elige 1, 2 o 3: ").strip()
        if choice in ("1", "2", "3"):
            return {"1": "humano", "2": "ia", "3": "azar"}[choice]
        print("Opcion no valida.")


def resolve_human_symbol(empieza, human_arg):
    """Devuelve el simbolo del humano ('X' u 'O').

    En este juego X siempre mueve primero, asi que quien empieza fija los simbolos:
    humano empieza -> humano X; ia empieza -> humano O.
    """
    if empieza is None:
        return human_arg or 'X'
    if empieza == 'azar':
        empieza = random.choice(['humano', 'ia'])
    wanted = 'X' if empieza == 'humano' else 'O'
    if human_arg is not None and human_arg != wanted:
        raise SystemExit(
            f"--human {human_arg} y --empieza {empieza} se contradicen: "
            f"X siempre mueve primero. Usa solo uno de los dos."
        )
    return wanted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--human', choices=['X', 'O'], default=None,
                    help='Simbolo del humano (X mueve primero).')
    ap.add_argument('--empieza', choices=['humano', 'ia', 'azar'], default=None,
                    help='Quien mueve primero. Sin --human ni --empieza se muestra un menu.')
    ap.add_argument('--simetrias', action='store_true',
                    help='la red promedia las 8 simetrias del tablero (1 forward con batch 8, sin busqueda)')
    args = ap.parse_args()

    if args.empieza is None and args.human is None and sys.stdin.isatty():
        args.empieza = ask_who_starts()
    args.human = resolve_human_symbol(args.empieza, args.human)

    path = args.model
    if not path.endswith('.pt'):
        path = os.path.join(
            os.path.dirname(__file__), '..', 'modelos', args.model + '.pt'
        )

    model, payload = load_checkpoint(path)

    # Verificacion defensiva del manifiesto del checkpoint.
    purity = payload.get("purity", {})
    required_false = (
        "mcts_used",
        "oracle_used_for_training",
        "search_used_for_training",
        "minimax_used_for_training",
        "teacher_used_for_training",
    )

    if any(purity.get(k, None) is not False for k in required_false):
        raise ValueError(
            "El checkpoint no tiene un manifiesto de entrenamiento puro verificable."
        )

    if purity.get("final_inference") != "one_forward_plus_legal_argmax":
        raise ValueError(
            "El checkpoint no declara inferencia pura por forward + argmax legal."
        )

    board = [EMPTY] * 25
    human = P1 if args.human == 'X' else P2
    player = P1

    print()
    print("======================================")
    print("       HUMANO VS RED NEURONAL PURA")
    print("======================================")
    print(f"Humano: {args.human}")
    print(f"IA: {'O' if args.human == 'X' else 'X'}")
    print(f"Empieza: {'Humano' if args.human == 'X' else 'IA'} (X mueve primero)")
    print()

    while True:
        show(board)

        done, winner = game_over(board)
        if done:
            if winner == 0:
                print("EMPATE")
            elif winner == human:
                print("GANASTE")
            else:
                print("GANO LA IA")
            return

        if player == human:
            print("TU TURNO")
            move = ask_human_move(board)
        else:
            print("IA pensando...")
            t0 = time.perf_counter()
            move = pure_move(model, board, player, args.simetrias)
            ms = (time.perf_counter() - t0) * 1000

            ai_row = move // 5 + 1
            ai_col = move % 5 + 1

            print(
                f"IA -> fila {ai_row}, columna {ai_col} "
                f"| {'1 forward CNN x 8 simetrias' if args.simetrias else '1 forward CNN'} | {ms:.2f} ms"
            )
            print()

        board = apply_move(board, move, player)
        player = 3 - player


if __name__ == '__main__':
    main()
