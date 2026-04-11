# Code Review — Token BiGRU fixes

1. Bug menor — Docstring desactualizado en el modelo

src/models/token_bigru/model.py:54

# Dice:
→ BiGRU(128) → [batch, 4096, 256]

# Ahora debería decir:
→ BiGRU(128, layers=2, dropout=0.3) → [batch, 4096, 256]

La descripción de la arquitectura todavía describe el modelo de 1 capa. Los shapes son correctos (la salida sigue
siendo [batch, 4096, 256]), pero quien lea el docstring no sabrá que hay 2 capas ni que hay dropout recurrente.

---
2. Bug menor — weight_decay no está en el docstring de train()

src/train/loop.py:284-293 — El bloque de "Parámetros" lista lr, epochs, patience, output_dir, pero weight_decay no
aparece. Los otros modelos que llamen a train() pasando weight_decay=0.0 (el default) no van a tener problema,
pero el parámetro está invisibilizado en la documentación.

---
3. Inconsistencia de trazabilidad — weight_decay no se imprime

src/train/main.py:525

Todos los parámetros clave se imprimen por pantalla:
Dispositivo: cuda
Muestras — train: 57036, ...
Modelo: 535404 parámetros
pos_weight base: 0.6299 → escalado: 0.9449
Pero weight_decay no aparece. Cuando revises el OUTPUT.txt de la próxima corrida no podrás confirmar el valor
usado a simple vista.

---
---
4. Detalle técnico — Adam L2 vs AdamW

src/train/loop.py:305

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

PyTorch's Adam con weight_decay aplica L2 antes del escalado adaptativo:
g_t = ∂L/∂w + λ·w       # L2 suma al gradiente
w_t+1 = w_t - α · Adam(g_t)  # Adam escala g_t adaptativamente

AdamW (Loshchilov & Hutter, 2019) aplica el decaimiento después, desacoplado del gradiente:
w_t+1 = (1 - λ) · w_t - α · Adam(∂L/∂w)

Con weight_decay=1e-4 la diferencia es prácticamente nula. Si en el futuro se sube el valor, torch.optim.AdamW
sería más correcto teóricamente.
