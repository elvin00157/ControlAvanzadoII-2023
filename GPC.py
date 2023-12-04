import numpy as np
import matplotlib.pyplot as plt

class GPCControlador:
    def __init__(self, A, B, H, p1, p2, Q, R, m):
        self.A = A
        self.B = B
        self.H = H
        self.p1 = p1
        self.p2 = p2
        self.Q = Q
        self.R = R
        self.m = m
        
        #Calculo de matrices G, F11 y F2
        self.G = self.calculo_G()
        self.F11 = self.calculo_F11()
        self.F2 = self.calculo_F2()
        self.g1 = self.calculo_g1()
    
    def calculo_G(self):
        G = np.zeros((self.p1, self.H.shape[0], self.A.shape[1]))
        for i in range(self.p1):
            G[i] = self.H @ np.linalg.matrix_power(self.A, i+1)
        return G.reshape((self.p1, -1))
    
    def calculo_F11(self):
        F11 = np.zeros((self.p1, self.p2))
        for i in range(self.p1):
            for j in range(min(i+1, self.p2)):
                #Ajuste de la potencia elevada de A
                potencia_A = i - j + (self.p1 - self.p2 - 1)
                F11[i, j] = self.H @ np.linalg.matrix_power(self.A, potencia_A) @ self.B
        return F11
    
    
    def calculo_F2(self):
        F2 = np.zeros((self.p1, self.B.shape[1]))
        for i in range(self.p1):
            sum_A = np.zeros_like(self.A)
            for k in range(i+1):
                sum_A += np.linalg.matrix_power(self.A, k)
            F2[i, :] = (self.H @ sum_A @ self.B).flatten()
        return F2
    
    def calculo_g1(self):
        #Calcular la matriz g1 basada en F11, Q y R
        F11_T_Q = self.F11.T @ self.Q
        g1 = np.linalg.inv(F11_T_Q @ self.F11 + self.R) @ F11_T_Q
        return g1[0:self.m, :]
    
    def calculo_senal_control(self, x_t, u_t_menos_1, Y_d):
        #Calcular la señal de control Δu(t)
        term1 = Y_d - self.G @ x_t - self.F2 @ u_t_menos_1
        DU = self.g1 @ term1
        return DU

Ds=0.35
#Valores para un F=175e-3
#Horizontes
p1 = 5 #Horizonte de predicción
p2 = 2 #Horizonte de control
m = 1 #Como sólo nos interesa el primer movimiento de control

#Definir las matrices de pesos Q y R.
valor_Q = 1.5
valor_R = 1

#Valores para un F=75e-3
#Horizontes
p1 = 3 #Horizonte de predicción
p2 = 3 #Horizonte de control
m = 1 #Como sólo nos interesa el primer movimiento de control
#Definir las matrices de pesos Q y R.
valor_Q = 0.3
valor_R = 0.1


Q = np.diag([valor_Q] * p1)  #Alto peso en el seguimiento de la referencia.
R = np.diag([valor_R] * p2)  #Peso reducido para permitir el esfuerzo de control.

# Parámetros iniciales
S0 = 59100000  # Condición inicial
E0 = 5 # Condición inicial
I0 = 3  # Condición inicial
L0 = 1  # Condición inicial
R0 = 1  # Condición inicial
t0 = 0  # Tiempo inicial
tf = 100  # Tiempo final
h = 1  # Tamaño de paso
beta=1.3
N=59100000
epsilon=1/4.3
landa=1/3.1
delta=1/33

#Matrices del espacio de estados
A_d = np.array([
    [0, 0, -beta, 0, 0],
    [0, -epsilon, beta, 0, 0],
    [0, epsilon, -landa, 0, 0],
    [0, 0, landa, -delta, 0],
    [0, 0, 0, delta, 0]
])
B_d = np.array([[1], [1], [1], [1], [1]])

H = np.array([[0, 0 , 1, 0, 0]])

#Tamaño del paso de tiempo
Delta_t = 0.1  #Ajustar segun la necesidad

#Matriz identidad del tamaño de A_d
I = np.eye(A_d.shape[0])

#Discretizar A_d y B_d
A = I + Delta_t * A_d
B = Delta_t * B_d

#Condiciones iniciales
x0 = np.array([S0, E0, I0, L0, R0])

#Trayectoria de referencia
Y_d = np.ones((p1, 1)) * 1000  #Punto de ajuste para la salida T

#Crear una instancia del controlador GPC con los parámetros dados
gpc_controlador = GPCControlador(A, B, H, p1, p2, Q, R, m)

#Número de pasos de simulación
mum_muestras = 365

#Inicializar el estado y controlar los vectores de entrada.
x_t = x0.reshape(-1, 1)  #Estado inicial
u_t_menos_1 = np.array([[Ds]])  #Entrada de control inicial (tasa de dilución de equilibrio)
Y = []  #Trayectoria de salida

#Simular el sistema con control GPC
for t in range(mum_muestras):
    #Calcular la señal de control.
    Du_t = gpc_controlador.calculo_senal_control(x_t, u_t_menos_1, Y_d)
    #Aplicar la señal de control al sistema (solo el primer elemento de ΔU)
    u_t = u_t_menos_1 + Du_t[0, 0]
    
    #Actualizar el estado del sistema
    x_t_plus_1 = A @ x_t + B * u_t
    
    #Calcular la salida
    y_t = H @ x_t
    
    #Almacenar la salida
    Y.append(y_t.item())
    
    #Actualizar variables para la próxima iteración
    x_t = x_t_plus_1
    u_t_menos_1 = np.array([[u_t]]).reshape(1, 1)


#Graficos en escala logaritmica
plt.figure(figsize=(12, 6))
plt.semilogy(range(mum_muestras), Y, label='COVID-19 (Caso Italia)')
plt.semilogy(range(mum_muestras), [1000]*mum_muestras, 'r--', label='Setpoint')
plt.title('Dinámica de los infectados (I) con Control GPC')
plt.xlabel('Tiempo [dias]')
plt.ylabel('Población Objetivo')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

#Inicializar el estado y controlar los vectores de entrada.
x_t = x0.reshape(-1, 1)  #Estado inicial
u_t = np.array([[Ds]])  #Entrada de control constante (tasa de dilución de equilibrio)
Y = []  #Trayectoria de salida

#Simular el sistema sin controlador
for t in range(mum_muestras):
    #Actualizar el estado del sistema
    x_t_plus_1 = A @ x_t + B * u_t

    #Calcular la salida
    y_t = H @ x_t

    #Almacenar la salida
    Y.append(y_t.item())

    #Actualizar variables para la próxima iteración
    x_t = x_t_plus_1

#Graficar la trayectoria de salida
plt.figure(figsize=(12, 6))
plt.semilogy(range(mum_muestras), Y, label='COVID-19 (Caso Italia)')
plt.title('Dinámica de los Infectados (I) (sin Control GPC)')
plt.xlabel('Tiempo [dias]')
plt.ylabel('Población')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()