from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

#Parámetros para el Reino Unido, usando los valores de la Tabla I
beta_0 = 1.28  #Tasa de transmisión inicial
gamma = 1/2.8  #Tasa de recuperación
epsilon = 1/6.2  #Tasa de incubación
delta = 0  #Tasa de mortalidad
tau_m = 10  #Retraso en días


#Parámetros de cambio para el Reino Unido según la Tabla I
t1 = 15
rho1 = 0.65
t2 = 25
rho2 = 0.27

#Tasa de transmisión que cambia en el tiempo
def beta(t, beta_0, t1, rho1, t2, rho2):
    if t < t1:
        return beta_0
    elif t < t2:
        return beta_0 * rho1
    else:
        return beta_0 * rho2

def covid_model(t, y, beta_0, t1, rho1, t2, rho2, gamma, epsilon, delta):
    E, I, L, T = y
    beta_t = beta(t, beta_0, t1, rho1, t2, rho2)
    
    dEdt = beta_t * I - epsilon * E
    dIdt = epsilon * E - gamma * I
    dLdt = gamma * I - delta * L
    dTdt = epsilon * E
    
    return [dEdt, dIdt, dLdt, dTdt]

#Función del modelo
#def covid_model(t, y):
#    E, I, L, T = y
#    beta = beta_0  #Aqui se puede implementar cualquier funcion de u(t) y d(t)
    
#    #Ecuaciones diferenciales
#    dEdt = beta * I - epsilon * E
#    dIdt = epsilon * E - gamma * I
#    dLdt = gamma * I - delta * L
#    dTdt = epsilon * E
    
#     return [dEdt, dIdt, dLdt, dTdt]

#Inicializar historia con condiciones iniciales
E0 = 1
I0 = 1
L0 = 0
T0 = 0
y0 = [E0, I0, L0, T0]

#Resolver el modelo
t_span = [0, 400]  # Un periodo que abarque suficiente tiempo
t_eval = np.linspace(t_span[0], t_span[1], int(t_span[1] - t_span[0]) + 1)  # Evaluar cada día

#sol = solve_ivp(covid_model, t_span, y0, t_eval=t_eval, dense_output=True)

#Resolver el modelo con parámetros que cambian en el tiempo
sol = solve_ivp(covid_model, t_span, y0, args=(beta_0, t1, rho1, t2, rho2, gamma, epsilon, delta), t_eval=t_eval, dense_output=True)

#Para los retrasos, usaremos la interpolación de la solución
#z = sol.sol

##Calculamos Nr, Ar y Tr con los retrasos
#Nr = epsilon * np.array([z(sol.t[i] - tau_m)[0] if sol.t[i] >= tau_m else E0 for i in range(len(sol.t))])
#Ar = np.array([z(sol.t[i] - tau_m)[1] + z(sol.t[i] - tau_m)[2] if sol.t[i] >= tau_m else I0 + L0 for i in range(len(sol.t))])
#Tr = np.array([z(sol.t[i] - tau_m)[3] if sol.t[i] >= tau_m else T0 for i in range(len(sol.t))])

#Interpolación para manejar retrasos
sol_interp = sol.sol

#Calculamos Nr, Ar y Tr con los retrasos
Nr = np.array([epsilon * sol_interp(sol.t[i] - tau_m)[0] if sol.t[i] >= tau_m else E0 for i in range(len(sol.t))])
Ar = np.array([sol_interp(sol.t[i] - tau_m)[1] + sol_interp(sol.t[i] - tau_m)[2] if sol.t[i] >= tau_m else I0 + L0 for i in range(len(sol.t))])
Tr = np.array([sol_interp(sol.t[i] - tau_m)[3] if sol.t[i] >= tau_m else T0 for i in range(len(sol.t))])

#Graficar Ar y Tr en escala logarítmica
plt.figure(figsize=(12, 6))
plt.plot(sol.t, Ar, label='Ar(t) - Casos Activos Reportados')
plt.plot(sol.t, Tr, label='Tr(t) - Total de Casos Reportados')
plt.title('Casos Activos y Totales Reportados acumulados')
plt.xlabel('Días desde el brote')
plt.ylabel('Número de individuos (escala logarítmica)')
plt.yscale('log')  #Establecer la escala del eje Y a logarítmica
plt.legend()
plt.grid(True)
plt.show()

#Graficar Nr en una gráfica separada en escala logarítmica
plt.figure(figsize=(12, 6))
plt.plot(sol.t, Nr, label='Nr(t) - Nuevos Casos Diarios Reportados')
plt.title('Nuevos Casos Diarios Reportados a lo largo del tiempo')
plt.xlabel('Días desde el brote')
plt.ylabel('Número de individuos (escala logarítmica)')
plt.yscale('log')  #Establecer la escala del eje Y a logarítmica
plt.legend()
plt.grid(True)
plt.show()

#Graficar E(t)
plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y[0], label='E(t) - Individuos Expuestos')
plt.title('Número de Individuos infectados, sin ser infecciosos')
plt.xlabel('Días desde el brote')
plt.ylabel('Número de Individuos Expuestos')
plt.legend()
plt.grid(True)
plt.show()

#Graficar I(t)
plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y[1], label='I(t) - Número de individuos infectados')
plt.title('Número de Individuos Infectados')
plt.xlabel('Días desde el brote')
plt.ylabel('Número de Individuos Infectados')
plt.legend()
plt.grid(True)
plt.show()

#Graficar L(t)
plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y[2], label='L(t)- Numero de individuos infectados Post-Latencia')
plt.title('Número de Individuos Post-Latencia')
plt.xlabel('Días desde el brote')
plt.ylabel('Número de Individuos Latentes')
plt.legend()
plt.grid(True)
plt.show()

#Graficar T(t)
plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y[3], label='T(t) - Total de Casos Detectados')
plt.title('Número Total de Casos Detectados a lo largo del tiempo')
plt.xlabel('Días desde el brote')
plt.ylabel('Número Total de Casos Detectados')
plt.legend()
plt.grid(True)
plt.show()