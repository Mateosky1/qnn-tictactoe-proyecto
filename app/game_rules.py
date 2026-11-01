"""Pure 5x5 / 4-in-a-row game rules for the no-search experiments."""
from dataclasses import dataclass, field

BOARD_N = 5
CELLS = 25
WIN_LEN = 4
EMPTY, P1, P2 = 0, 1, 2
DIRS = ((0,1),(1,0),(1,1),(1,-1))

_LINES=[]
for r in range(BOARD_N):
    for c in range(BOARD_N):
        for dr,dc in DIRS:
            rr,cc=r+dr*(WIN_LEN-1),c+dc*(WIN_LEN-1)
            if 0<=rr<BOARD_N and 0<=cc<BOARD_N:
                _LINES.append(tuple((r+dr*k)*BOARD_N+(c+dc*k) for k in range(WIN_LEN)))
LINES=tuple(_LINES)

def legal_moves(board):
    return [i for i in range(CELLS) if board[i] == EMPTY]

def apply_move(board,i,player):
    if i < 0 or i >= CELLS or board[i] != EMPTY:
        raise ValueError(f"illegal move: {i}")
    b=list(board); b[i]=player; return b

def winning_line(board,player):
    for line in LINES:
        if all(board[i] == player for i in line): return line
    return None

def has_won(board,player): return winning_line(board,player) is not None

def is_full(board): return EMPTY not in board

def game_over(board):
    if has_won(board,P1): return True,P1
    if has_won(board,P2): return True,P2
    if is_full(board): return True,0
    return False,None

def immediate_wins(board,player):
    return [i for i in legal_moves(board) if has_won(apply_move(board,i,player),player)]

@dataclass
class MoveReport:
    move:int=-1
    value:float=0.0
    candidates:list=field(default_factory=list)
    forwards:int=0
    nodes:int=0
    depth:int=0
    sims:int=0
    elapsed_ms:float=0.0
    reason:str=""
    activations:dict=None
    extra:dict=field(default_factory=dict)
