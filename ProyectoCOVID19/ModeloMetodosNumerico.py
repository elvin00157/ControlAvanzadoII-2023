import numpy as np
import matplotlib.pyplot as plt

def edoS(y, t, beta,I,S,N,E,L,R,epsilon, landa,delta):
    return -beta*I*y/N

def edoE(y, t, beta,I,S,N,E,L,R,epsilon, landa,delta):
    return beta*I*S/N-epsilon*y

def edoI(y, t, beta,I,S,N,E,L,R,epsilon, landa,delta):
    return  epsilon*E-landa*y

def edoL(y, t, beta,I,S,N,E,L,R,epsilon, landa, delta):
    return  landa*I-delta*y

def edoR(y, t, beta,I,S,N,E,L,R,epsilon, landa,delta):
    return  delta*L



def runge_kutta_cuarto_orden(fS,fE,fI,fL,fR, S0,E0,I0,L0,R0, t0, tf, h, beta,N,epsilon, landa,delta):
    # Implementa el método de Runge-Kutta de cuarto orden
    tiempo = np.arange(t0, tf + h, h)
    resS = []
    resE = []
    resI = []
    resL = []
    resR = []

    S_actual = S0
    E_actual = E0
    I_actual = I0
    L_actual = L0
    R_actual = R0
    for t in tiempo:
        resS.append(S_actual)
        k1 = h * fS(S_actual, t, beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k2 = h * fS(S_actual + 0.5 * k1, t + 0.5 * h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k3 = h * fS(S_actual + 0.5 * k2, t + 0.5 * h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k4 = h * fS(S_actual + k3, t + h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        S_actual = S_actual + (k1 + 2*k2 + 2*k3 + k4) / 6

        resE.append(E_actual)
        k1 = h * fE(E_actual, t,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k2 = h * fE(E_actual + 0.5 * k1, t + 0.5 * h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k3 = h * fE(E_actual + 0.5 * k2, t + 0.5 * h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k4 = h * fE(E_actual + k3, t + h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        E_actual = E_actual + (k1 + 2*k2 + 2*k3 + k4) / 6

        resI.append(I_actual)
        k1 = h * fI(I_actual, t, beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k2 = h * fI(I_actual + 0.5 * k1, t + 0.5 * h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k3 = h * fI(I_actual + 0.5 * k2, t + 0.5 * h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k4 = h * fI(I_actual + k3, t + h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        I_actual = I_actual + (k1 + 2*k2 + 2*k3 + k4) / 6

        resL.append(L_actual)
        k1 = h * fL(L_actual, t,beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k2 = h * fL(L_actual + 0.5 * k1, t + 0.5 * h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k3 = h * fL(L_actual + 0.5 * k2, t + 0.5 * h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k4 = h * fL(L_actual + k3, t + h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        L_actual = L_actual + (k1 + 2*k2 + 2*k3 + k4) / 6

        resR.append(R_actual)
        k1 = h * fR(R_actual, t, beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k2 = h * fR(R_actual + 0.5 * k1, t + 0.5 * h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k3 = h * fR(R_actual + 0.5 * k2, t + 0.5 * h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        k4 = h * fR(R_actual + k3, t + h,  beta,I_actual,S_actual,N,E_actual,L_actual, R_actual, epsilon, landa,delta)
        R_actual = R_actual + (k1 + 2*k2 + 2*k3 + k4) / 6

    return tiempo, np.array(resS),np.array(resE),np.array(resI),np.array(resL),np.array(resR)

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

# Resuelve la ecuación diferencial utilizando Runge-Kutta de cuarto orden
tiempo, S, E, I,L,R = runge_kutta_cuarto_orden(edoS,edoE,edoI,edoL,edoR, S0,E0,I0,L0,R0, t0, tf, h, beta,N,epsilon, landa,delta)

# Grafica los resultados
plt.plot(tiempo, E, label='E: Expuestos, la tienen No son infecciosos')
plt.plot(tiempo, S, label='S: Susceptibles')
plt.plot(tiempo, I, label='I: Infectados')
plt.plot(tiempo, L, label='L: Ya no son infecciosos')
plt.plot(tiempo, R, label='R: Recuperados')
plt.xlabel('Tiempo')
plt.ylabel('y(t)')
plt.title('Comportamiento del covid 19 en Italia en febrero a junio')
plt.legend()
plt.grid(True)
plt.show()
