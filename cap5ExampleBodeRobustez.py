import control as ctrl
import matplotlib.pyplot as plt
import numpy as np

# Define la función de transferencia en lazo cerrado
numerator = [0.5] # Los coeficientes del numerador
denominator = [1, 2, 1, 0]  # los coeficientes del denominador
system = ctrl.TransferFunction(numerator, denominator)
plt.figure()
plt.title('Diagrama de Bode - Magnitud')
# Calcula el diagrama de Bode
frequency, magnitude, phase = ctrl.bode(system, dB=True)

# Calcula el margen de fase y la frecuencia de cruce
gain_margin, phase_margin, _, crossover_frequency = ctrl.margin(system)

phase_margin_radians = np.radians(phase_margin)
# Muestra los resultados
print(f'Margen de Ganancia: {gain_margin} dB')
print(f'Margen de Fase: {phase_margin_radians} rad')
print(f'Margen de Fase: {phase_margin} grados')
print(f'Frecuencia de P: {crossover_frequency} rad/s')
print(f'El tau calculado en el inciso a es: {phase_margin_radians/crossover_frequency} rad/s')

# Agregando una función compleja para graficar variando tau
tau=1
complex_function1 = lambda x: np.abs(np.exp(-1j*x*tau)-1)*np.abs((0.5)/(-x*x*x*1j-2*x*x+1j*x+0.5))

tau2=1.5
complex_function2 = lambda x: np.abs(np.exp(-1j*x*tau2)-1)*np.abs((0.5)/(-x*x*x*1j-2*x*x+1j*x+0.5))

tau3=2
complex_function3 = lambda x: np.abs(np.exp(-1j*x*tau3)-1)*np.abs((0.5)/(-x*x*x*1j-2*x*x+1j*x+0.5))

x_values = np.logspace(-1, 1, 100)
y_values1 = complex_function1(x_values)
plt.figure()
plt.grid(True)
plt.plot(x_values, y_values1, label='T=1' )
plt.title('Comportamiento cota Robustez variando el tau con un G real')
plt.xlabel("Magnitud [db]")
plt.ylabel("Frecuencia [rad/s]")
y_values2 = complex_function2(x_values)
plt.plot(x_values, y_values2, label='T=1.5')
y_values3 = complex_function3(x_values)
plt.plot(x_values, y_values3, label='T=2')
plt.legend()
plt.show()
