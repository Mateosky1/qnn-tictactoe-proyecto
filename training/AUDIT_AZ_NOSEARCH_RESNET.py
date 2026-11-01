"""Auditoria de pureza: ningun archivo de entrenamiento ni de inferencia importa o llama
MCTS, minimax, alpha-beta, Oracle, busqueda ni un maestro. (evaluacion/ queda fuera a proposito:
ahi vive el Oracle, que solo se usa para medir.)"""
import ast, os
ROOT=os.path.dirname(os.path.abspath(__file__))
FILES=[  # todo el codigo que entrena o que decide la jugada de la red
 os.path.join(ROOT,"train_az_nosearch_v23.py"),
 os.path.join(ROOT,"selfplay_vec.py"),
 os.path.join(ROOT,"..","app","az_resnet_model.py"),
 os.path.join(ROOT,"..","app","inferencia.py"),
 os.path.join(ROOT,"..","app","jugar_az_nosearch_resnet.py"),
 os.path.join(ROOT,"..","app","game_rules.py"),
]
BAD=("mcts","minimax","negamax","alpha-beta","alphabeta","oracle","search","teacher")
h=[]
for p in FILES:
 t=open(p,encoding="utf8").read(); tree=ast.parse(t,p)
 for n in ast.walk(tree):
  if isinstance(n,(ast.Import,ast.ImportFrom)):
   s=ast.unparse(n).lower()
   if any(x in s for x in BAD): h.append((p,"import",s))
  elif isinstance(n,ast.Call):
   s=ast.unparse(n.func).lower()
   if any(x in s for x in BAD): h.append((p,"call",s))
print("FILES_CHECKED",len(FILES)); print("FORBIDDEN_EXECUTABLE_REFERENCES",h); print("PASS" if not h else "FAIL")
