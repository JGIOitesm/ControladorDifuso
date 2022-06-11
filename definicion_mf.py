import numpy as np
import csv
import skfuzzy as sk
from matplotlib import pyplot as plt

dist_t2 = np.arange (0, 16, 0.1)
angle_t2 = np.arange (-180, 180, 0.5)
vel_t1 = np.arange (0, 4, 0.1)
w_t1 = np.arange(-2,2, 0.1 )

# Distances to Turtle 2 => (Close, Medium Close, Medium, Medium Far, Far)

d_t2_mfs =  [sk.gaussmf(dist_t2,16, 1.5),    # Far
            sk.gaussmf(dist_t2,12, 1.5),     # Medium Far
            sk.gaussmf(dist_t2,8, 1.5),      # Medium 
            sk.gaussmf(dist_t2,4, 1.5),      #Medium Close
            sk.gaussmf(dist_t2, 0, 1.5)]     #Close

# Angle difference between Turtle1 & Turtle 2 => (Extreme Left, Medium Left, Centered)

a_t2_mfs =  [sk.gaussmf(angle_t2, 180, 35), # Extreme Left
            sk.gaussmf(angle_t2, 90, 35), # Medium Left
            sk.gaussmf(angle_t2, 0, 35),  # Centered
            sk.gaussmf(angle_t2, -90, 35),    # Medium Right
            sk.gaussmf(angle_t2, -180, 35)]   # Extreme Right

# Output Velocidad Lineal

v_t1_mfs = [sk.trimf(vel_t1, [3,4,4]),      # Alta
            sk.trimf(vel_t1, [2,3,4]),      # Media-Alta
            sk.trimf(vel_t1, [1, 2, 3]),    # Media
            sk.trimf(vel_t1, [0,1,2]),      # Media-Baja
            sk.trimf(vel_t1, [0,0,1])]      # Baja

# Output Velocidad Angular

w_t1_mfs = [sk.trimf(w_t1, [1,2,2]),      # Alta Izquierda
            sk.trimf(w_t1, [0,1,2]),      # Media-Izquierda
            sk.trimf(w_t1, [-1, 0, 1]),   # Baja
            sk.trimf(w_t1, [-2,-1,0]),    # Media-Derecha
            sk.trimf(w_t1, [-2,-2,-1])]   # Alta Derecha

def turtle_calc(dist_input, angle_input, show = False):
    dist_indx = round((dist_input)/0.1)
    angle_indx = round((angle_input + 180)/0.5)
    
    # Reglas de inferencia
    R = [[min(d_t2_mf[dist_indx], a_t2_mf[angle_indx]) for a_t2_mf in a_t2_mfs]for d_t2_mf in d_t2_mfs]

    v_cut_t1 = [R[0][2],                                                                        # Alta => dist_low y vel_med, dist_low y vel_high
                max(R[0][1],R[0][3],R[1][2]),                                                   # Media-alta => dist_low y vel_low, dist_med y vel_high
                max(R[0][0],R[0][4],R[1][1],R[1][3],R[2][2]),                              # Media => dist_med y vel_med
                max(R[1][0],R[1][4],R[2][0],R[2][1],R[2][3],R[2][4],R[3][2]),                           # Media-baja => dist_med y vel_low, dist_high y vel_high
                max(R[4][2],R[3][0],R[3][1],R[3][3],R[3][4],R[4][0],R[4][1],R[4][3],R[4][4])]  # Baja => dist_high y vel_low, dist_high y vel_med

    w_cut_t1 = [max(R[2][0],R[3][0],R[4][0]),                   # Alta Izquierda => dist_low y vel_med, dist_low y vel_high
                max(R[0][0],R[0][1],R[1][0],R[1][1],R[2][1],R[3][1],R[4][1]),   # Media Izquierda => dist_low y vel_low, dist_med y vel_high
                max(R[0][2],R[1][2],R[2][2],R[3][2],R[4][2]),                   # Baja => dist_med y vel_med
                max(R[0][3],R[0][4],R[1][3],R[1][4],R[2][3],R[3][3],R[4][3]),           # Media Derecha => dist_med y vel_low, dist_high y vel_high
                max(R[2][4], R[3][4],R[4][4])]          # Alta Derecha => dist_high y vel_low, dist_high y vel_med

    velocities_output = np.zeros(vel_t1.shape)
    omega_output = np.zeros(w_t1.shape)
    xy_sum_v = 0    # For linear velocity
    area_v = 0
    xy_sum_w = 0    # For angular velocity
    area_w = 0
    for i,vel in enumerate(vel_t1):
        velocities_output[i] = max([min(v_cut_t1[mf_index], v_t1_mfs[mf_index][i]) for mf_index in range(len(v_cut_t1))])
        xy_sum_v += vel*velocities_output[i]
        area_v += velocities_output[i]

    for i,omega in enumerate(w_t1):
        omega_output[i] = max([min(w_cut_t1[mf_index], w_t1_mfs[mf_index][i]) for mf_index in range(len(w_cut_t1))])
        xy_sum_w += omega*omega_output[i]
        area_w += omega_output[i]

    center_v = xy_sum_v/area_v # Velocidad a la que debe cruzar
    center_w = xy_sum_w/area_w # Velocidad a la que debe cruzar

    return (center_v, velocities_output, center_w, omega_output) if show else (center_v, center_w)

dist_input = round(float(input("Distancia de la tortuga: ")),1)
angle_input = round(float(input("Angulo de la tortuga: ")),1)
center_v, vel_t1_output, center_w, omega_t1_output = turtle_calc(dist_input, angle_input, True)
print("Velocidad lineal de la tortuga: ", center_v)
print("Velocidad angular de la tortuga: ", center_w)

with plt.xkcd():
    plt.subplot(2,3,1)
    for dist_mf in d_t2_mfs:
        plt.plot(dist_t2, dist_mf)
    plt.legend(["Lejos", "Medio lejos","Medio", "Medio cerca", "Cerca"], loc='lower left')
    plt.ylabel(r'$\mu$')
    plt.xlabel("m")
    plt.title("Entrada 1: Distancia de la tortuga 2")

    plt.subplot(2,3,4)
    for angle_mf in a_t2_mfs:
        plt.plot(angle_t2, angle_mf)
    plt.legend(["Extremo izquierda", "Medio izquierda", "Centrado", "Medio derecja", "Extremo derecha"], loc='lower left')
    plt.ylabel(r"$\mu$")
    plt.xlabel(r"$\alpha$")
    plt.title('Entrada 2: Diferencia de ángulos')

    plt.subplot(2,3,2)
    for v_t1_mf in v_t1_mfs:
        plt.plot(vel_t1, v_t1_mf)
    plt.legend(["Alta", "Media alta", "Media", "Media baja", "Baja"], loc='lower left')
    plt.ylabel(r'$\mu$')
    plt.xlabel('m/s')
    plt.title('Salida: Velocidad Lineal Tortuga 1')

    plt.subplot(2,3,5)
    plt.plot(vel_t1, vel_t1_output)
    plt.ylabel(r'$\mu$')
    plt.xlabel('m/s')
    plt.ylim([0, 1.05])
    plt.title('Control Velocidad Lineal')

    plt.subplot(2,3,3)
    for w_t1_mf in w_t1_mfs:
        plt.plot(w_t1, w_t1_mf)
    plt.legend(["Alta Derecha", "Media Derecha", "Baja", "Media Izquierda", "Alta Izquirda"], loc='lower left')
    plt.ylabel(r'$\mu$')
    plt.xlabel('m/s')
    plt.title('Salida: Velocidad Angular Tortuga 1')

    plt.subplot(2,3,6)
    plt.plot(w_t1, omega_t1_output)
    plt.ylabel(r'$\mu$')
    plt.xlabel('rad/s')
    plt.ylim([0, 1.05])
    plt.title('Control Velocidad Angular')
    plt.show()

plt.figure(1)
ax = plt.axes(projection = '3d')
X, Y = np.meshgrid(angle_t2, dist_t2)
Z1 = np.zeros((len(dist_t2), len(angle_t2)))
Z2 = np.zeros((len(dist_t2), len(angle_t2)))
for angle in range(len(angle_t2)):
    for dist in range(len(dist_t2)):
        Z1[dist,angle],Z2[dist,angle] = turtle_calc(dist_t2[dist], angle_t2[angle])


        
with open("surface_v_lin.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(Z1)
with open("surface_v_ang.csv", "w") as f:
    wr = csv.writer(f)
    wr.writerows(Z2)

ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap='inferno', edgecolor='none')
ax.set_xlabel("Angulo tortuga [rad]")
ax.set_ylabel("Distancia tortuga [m]")
ax.set_zlabel("Velocidad Lineal [m/s]")
ax.set_title("Superficie de Control Velocidad Lineal")
plt.show()


plt.figure(2)
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap='inferno', edgecolor='none')
ax.set_xlabel("Angulo tortuga [rad]")
ax.set_ylabel("Distancia tortuga [m]")
ax.set_zlabel("Velocidad Angular [rad/s]")
ax.set_title("Superficie de Control Velocidad Angular")
plt.show()
