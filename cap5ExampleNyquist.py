import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Aqui  definimos la funcion de transferencia, agregando unicamente los coeficientes
numerator = [1]
denominator = [-1, -2,-3,0]

# Creamos un sistema de control
system = ctrl.TransferFunction(numerator, denominator)

# Calcula el diagrama de Nyquist
G = ctrl.zpk([-2], [-1, -3], gain=1)

fig, ax = plt.subplots()
out = ctrl.nyquist_plot(G)
# Gráfica el diagrama de Nyquist

ax.plot(out, label='Nyquist', linewidth=1, color="blue")

# Personaliza el estilo de la gráfica de Nyquist
ax.grid(True)
ax.set_title("Diagrama de Nyquist para: FTba=(s-2)/(s+1)(s+3)")
ax.set_xlabel("Parte Real")
ax.set_ylabel("Parte Imaginaria")

# Marca el punto crítico
ax.plot(-1, 0, 'ro', markersize=6, label='Punto Critico')

# Personaliza la leyenda
ax.legend()

plt.show()

