# Red neuronal pura para 5x5 con 4 en raya

Una CNN residual aprende a jugar solo con partidas contra sí misma (self-play + PPO).
**No usa búsqueda de ningún tipo** (ni MCTS, ni minimax, ni el Oracle) para entrenar ni para jugar.
El Oracle (minimax exacto con límite de tiempo) se usa únicamente para medir a la red.

La documentación completa (código, red, entrenamiento, función de pérdida, resultados y gráficas)
está en `docs/DOCUMENTACION.docx`.

## Estructura

```
app/
  game_rules.py                 reglas del 5x5 con 4 en raya
  az_resnet_model.py            la red (CNN residual, política + valor)
  inferencia.py                 cómo elige jugada la red: pura (1 forward) o con simetrías
  jugar_az_nosearch_resnet.py   jugar contra la red en consola
training/
  train_az_nosearch_v23.py      entrenamiento (self-play + PPO/GAE)
  selfplay_vec.py               motor de self-play vectorizado + GAE
  test_selfplay_vec.py          pruebas del motor
  AUDIT_AZ_NOSEARCH_RESNET.py   auditoría de pureza (entrenamiento e inferencia sin búsqueda)
evaluacion/
  oracle_minimax.py             el Oracle (solo para evaluar)
  partidas_vs_oracle.py         1000 partidas por enfrentamiento, en paralelo y reanudable
  graficas.py                   todas las gráficas
interfaz/
  app_gui.py                    interfaz grafica: red vs Oracle, tu vs red, resultados
docs/
  generar_documentacion.py      genera docs/DOCUMENTACION.docx
  DOCUMENTACION.docx
modelos/
  largo1_gen500.pt              primer checkpoint (update 500)
  largo1_final.pt               último checkpoint (update 6466)
resultados/
  entrenamiento/                log_largo1.txt, history.json, purity_manifest.json
  partidas/                     resultados contra el Oracle (se llena al correr las partidas)
  graficas/                     gráficas generadas
```

## Instalación

```
pip install torch numpy matplotlib python-docx
```

## Pasos

Windows (PowerShell) usa `cd ..\docs`; Mac/Linux usa `cd ../docs`.

**1. Verificar el código**
```
cd training
python test_selfplay_vec.py
python AUDIT_AZ_NOSEARCH_RESNET.py
```

**2. Entrenamiento** (solo si quieres repetirlo; `largo1` ya está entrenado con esto)
```
cd training
python train_az_nosearch_v23.py --preset full --games 512 --minutes 90 --seed 1 --eval-every 500 --out largo1
```

**3. Partidas contra el Oracle**: primer y último checkpoint × Oracle 2/4/10 s × 1000 partidas
```
cd evaluacion
python partidas_vs_oracle.py --procesos 6
```
- `--procesos`: usa como máximo uno por núcleo libre. Si los procesos compiten por la CPU, el Oracle piensa menos dentro de su tiempo.
- Es **reanudable**: si se corta, vuelve a correr el mismo comando y sigue donde quedó.
- Por defecto la red juega en modo `simetrias`. Para el modo puro: `--modo pura`.
- Prueba corta antes de lo largo: `python partidas_vs_oracle.py --modelos largo1_final --tiempos 2 --partidas 20`

**4. Gráficas**
```
cd evaluacion
python graficas.py
```

**5. Documento Word con los resultados**
```
cd docs
python generar_documentacion.py
```

**6. Jugar contra la red**
```
cd app
python jugar_az_nosearch_resnet.py --model largo1_final --simetrias
```

**7. Interfaz grafica**
```
python interfaz/app_gui.py
```
- *Red vs Oracle*: elige 2/4/10 s, checkpoint, modo de la red, quien empieza y jugadas al azar.
  Muestra en vivo lo que piensa la red (mapa de calor), su evaluacion, los nodos y el tiempo del
  Oracle (y si resolvio exacto o se quedo sin tiempo), graficas de la partida y el marcador de la sesion.
  Tambien juega series de 10 partidas y guarda partidas en `resultados/partidas_interfaz/`.
- *Tu vs Red*: haz clic para jugar; puedes ver lo que jugaria la red y deshacer.
- *Resultados*: las 1000 partidas por enfrentamiento y las curvas del entrenamiento.
- Atajos: Espacio jugar/pausa, N siguiente jugada, R reiniciar, 1/2/3 Oracle 2/4/10 s, H mapa de calor, U deshacer.
- Usa tkinter (viene con Python). En Mac con Python de Homebrew: `brew install python-tk`.

## Modo `--simetrias` (no es búsqueda)

La misma red recibe en **un solo forward** (lote de 8) el tablero en sus 8 rotaciones y reflejos.
Cada salida se devuelve a la orientación real y se promedian las log-probabilidades.
No se simula ninguna jugada ni se mira el futuro. Sin la opción, la red juega con 1 forward + argmax legal.
