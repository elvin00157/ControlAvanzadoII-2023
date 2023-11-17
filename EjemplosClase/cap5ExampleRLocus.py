import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Define la función de transferencia. Ajusta los coeficientes según tu sistema.
numerator = [1]
denominator = [1, 3, 2,0]

# Calcula el Lugar de las Raíces
system = ctrl.TransferFunction(numerator, denominator)
K_values = np.linspace(0, 10, 1000)
roots, gains = ctrl.root_locus(system, kvect=K_values, PrintGain=False, Plot=False)

# Gráfica el Lugar de las Raíces con estilo personalizado
print(system)
fig, ax = plt.subplots()
ax.grid(True)
ax.set_title("Lugar de las Raíces para G(s)H(s)=k/(s(s+1)(s+2)) ")
ax.set_xlabel("Parte Real")
ax.set_ylabel("Parte Imaginaria")

for i in range(roots.shape[1]):
    ax.plot(np.real(roots[:, i]), np.imag(roots[:, i]), label=f'K = Tramo {i+1}', linestyle='-', linewidth=2)

# Destacar el punto crítico
ax.plot(0, 0, 'rx', markersize=10, label='Polo s=0')
ax.plot(-1, 0, 'rx', markersize=10, label='Polo s=-1')
ax.plot(-2, 0, 'rx', markersize=10, label='Polo s=-2')
# Mostrar líneas de referencia
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.axvline(0, color='k', linestyle='--', linewidth=0.5)

# Personalizar la leyenda
ax.legend(loc='best', fontsize='small')

plt.show()
