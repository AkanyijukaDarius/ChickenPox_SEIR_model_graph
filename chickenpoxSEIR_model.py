#akanyijukadarius
#dariusakanyijuka3@gmail.com
#github: https://github.com/AkanyijukaDarius

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#parameters
beta = 0.4 #infection rate
mu = 0.001 #Natural death rate
delta = 0.2 #rate of progression from exposed to infected
gamma = 0.3 # Recovery rate
d = 0.0005 #disease induced death rate
Lambda = 1.0 #Recruitment rate(assumed constant

# total population
N = 1000000 #Population size(1 million)
S0 = 0.99 * N #susceptible population (99% of N)
E0 = 0.001 * N #exposed population (0.1% of N)
I0 = 0.001 * N # infected population (0.1% of N)
R0 = 0.0 # Recovered population(initially 0)

#initial conditions
y0 = [S0, E0, I0, R0]

#system of differential equations
def model(y,t):
    S,E,I,R =y
    dSdt = Lambda - beta*S*I/N - mu*S
    dEdt = beta*S*I/N - (delta+mu)*E
    dIdt = delta*E - (gamma+mu+d)*I
    dRdt = gamma*I - mu*R
    return [dSdt, dEdt, dIdt, dRdt]

#time points (from 0 to 200days)
t =np.linspace(0,200,200)

#solving the system of equations
solution = odeint(model,y0,t)

#the results
S,I,E,R = solution.T

#ploting the results
plt.figure(figsize=(10,6))
plt.plot(t,S/N,label='Susceptible',color='blue')
plt.plot(t,E/N,label='Exposed',color='Orange')
plt.plot(t,I/N,label='Infected',color='red')
plt.plot(t,R/N,label='Recovered',color='green')
plt.xlabel('Time(days)')
plt.ylabel('Proportion of the population')
plt.title('SEIR Model Simulation (with realistic population size)')
plt.legend()
plt.grid(True)
plt.show()



