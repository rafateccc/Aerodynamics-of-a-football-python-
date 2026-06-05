import numpy as np
import matplotlib.pyplot as plt
import os
import ezc3d
#Variable definitions 

#Ball variables & constants 
w=np.array([-8.135,2.455,25.391]) #Omega definition  rad/s (Data from a specific shot)
w_hat=w/np.sqrt(w[0]**2 + w[1]**2 + w[2]**2)

R=69/(2*np.pi) *10**(-2)#Ball radius (meters)
d=2*R  #Diameter of the ball (meters)
A= np.pi*R**2#Cross sectional area of the sphere m^2
m=0.43 #Ball's mass (kg)



#Aerodynamic variables & constants 
mu=1.82*10**(-5) #Air viscosity in Pa*s at 20ºC
rho=1.204 #Air density at 20ºC (kg/m^3)

#Movement equations

g=9.81 #Acceleration due to gravity (m/s^2)
Fw=np.array([0.0,0.0,-m*g]) #Force due to weight of the ball (Newtons) 
position = np.array([0.480111633, -0.125103348, 0.145390839])  # [x,y,z] (m) z=vertical direction, y=side direction, x=principal direction of the shot
velocity = np.array([28.23, 4.82, 5.25]) # [vx, vy, vz] m/s # values from a measure in our lab (see algo line 130)

#All these parameters are obtained from telstar_parameters.py
v_min_fixed=13.756801532109767 
a      = 0.5677 
b_min  = 0.1592 
v_c    = 9.9346 
v_s    = 1.1298 
p   = 0.0012 #tail slope

#Obtained from Kiratidis and Leinweber (2018) for recent FIFA World Cup balls.
Cd_brazuca_30 =0.2000425746
Cd_telstar_30 = 0.1731581914 
Cd_teamgeist_30=0.1497636901
b_teamgeist=0.514
b_telstar = b_teamgeist * ( 1 + 0.05 * (Cd_telstar_30 - Cd_teamgeist_30)/(Cd_brazuca_30 - Cd_teamgeist_30))


def Cd(v,Sp): #Drag coeff
    term1 = (a - b_min) / (1 + np.exp((v - v_c) / v_s))
    term2 = b_min
    term3 = ((v - v_min_fixed) / (1 + np.exp(-(v - v_min_fixed) / v_s))) * p
    Cd_0=term1 +term2 +term3 
    value=Cd_0+b_telstar*Sp  
    return value

def Fd(v,vhat,Cd_val): #Drag force (vector)
    Fd=(-(rho*Cd_val*A*v**2)/2)*vhat #pointing towards vhat (vhat in cartesian coord)
    return Fd

def Cm(v,Sp): #Magnus coefficient 
    Cm_fit= 1.15*Sp**0.83
    Cd_Re_0=Cd(0,0) 
    Cm=Cm_fit*(Cd_Re_0-np.minimum(Cd(v,Sp),Cd_Re_0))/(Cd_Re_0-Cd(22.97,0.19))
    return Cm
    
def Fm(velocity, Cm_val):  # Magnus force pointing towards w x v
    v = np.linalg.norm(velocity)
    if v == 0:
        return np.zeros(3)
    direction = np.cross(w, velocity)
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        return np.zeros(3)
    lhat = direction / direction_norm
    return (rho * Cm_val * A * v**2 / 2) * lhat

def RK4(position,velocity,time,dt):
    def f (t,Y):
        x, y, z, vx, vy, vz = Y
        velocity=np.array([vx,vy,vz])
        v=np.linalg.norm(velocity) #Module of the linear velocity
        vhat=velocity/v #Unitary vector pointing towards velocity 
        w_parallel = np.dot(w, vhat) * vhat
        w_perp = w - w_parallel
        Sp = (R*np.linalg.norm(w_perp))/v  #Perpendicular spin parameter
        F_drag = Fd(v,vhat,Cd(v,Sp))
        F_magnus = Fm(velocity,Cm(v,Sp))
        F_total=F_drag+F_magnus+Fw
        ax,ay,az=F_total/m
        return np.array([velocity[0], velocity[1],velocity[2], ax, ay, az])
    steps=int(time/dt)
    Sp_values=[]
    v_values=[]
    Cd_values=[]
    Cm_values=[]
    Re_values=[]
    vx_values = []
    vy_values = []
    vz_values = []
    x_values=[]
    y_values=[]
    z_values=[]
    t_values=[]
    total_time=0
    Y= np.array([position[0],position[1],position[2],velocity[0],velocity[1],velocity[2]],dtype=float) 
    for i in range(steps+1): 
        x_values.append(Y[0])
        y_values.append(Y[1])
        z_values.append(Y[2])
        vx_values.append(Y[3])
        vy_values.append(Y[4])
        vz_values.append(Y[5])
        t_values.append(total_time)
        velocity_current = Y[3:6]
        v = np.linalg.norm(velocity_current)
        vhat = velocity_current/v

        w_parallel = np.dot(w, vhat) * vhat
        w_perp = w - w_parallel

        Sp = (R*np.linalg.norm(w_perp))/v  #Perpendicular spin parameter
        Re = (rho*v*d)/mu #Reynolds number (adimensional)

        Sp_values.append(Sp)
        Re_values.append(Re)
        Cd_values.append(Cd(v,Sp))
        Cm_values.append(Cm(v,Sp))
        v_values.append(v)
        k1=f(total_time,Y)
        k2=f(total_time+dt/2,Y+(k1*dt/2))
        k3=f(total_time+dt/2,Y+k2*dt/2)
        k4=f(total_time+dt,Y+k3*dt)
        Y+=(dt/6)*(k1+2*k2+2*k3+k4)
        total_time+=dt
    return  x_values, y_values, z_values, t_values, Re_values, Sp_values, v_values, Cd_values, Cm_values, vx_values, vy_values, vz_values
x_values, y_values, z_values, t_values, Re_values, Sp_values, v_values, Cd_values, Cm_values, vx_values, vy_values, vz_values=RK4(position,velocity,0.040,0.004) #all lists of values

#Vicon measurements for a specific shot
x_vicon=[0.480111633, 0.592282959 ,0.704390564 ,0.815973267 ,0.926757629 ,1.038693604 ,1.152937988 ,1.261086914 ,1.369974854 ,1.480855469 ,1.591247803]
y_vicon=[-0.125103348, -0.106498337, -0.087933426, -0.069072792, -0.052011436, -0.032219078, -0.014402132, 0.003731833, 0.022838366, 0.040374619, 0.05822888]
z_vicon=[0.145390839, 0.166156555,0.186319824, 0.206325958, 0.225630280, 0.246114777, 0.263035583, 0.282501404, 0.302564087,0.321324402,0.339626770]
vx_vicon = [28.235914, 28.079268, 27.976593, 27.760956, 27.775324, 28.423515, 27.831850, 26.961115, 27.496656]
vy_vicon = [4.821163, 4.633962, 4.714842, 4.439513, 4.610384, 4.751454, 4.432448, 4.694381, 4.598276]
vz_vicon = [5.248859, 5.136384, 5.023245, 4.885946, 5.033226, 4.647227, 4.461653, 5.021214, 4.874850]

# x,y,z simulation vs vicon
# x, y, z simulation vs Vicon in one single figure
variables = [
    ("x", x_values, x_vicon, "position [m]"),
    ("y", y_values, y_vicon, "position [m]"),
    ("z", z_values, z_vicon, "position [m]"),
]

# x, y, z simulation vs Vicon in one single graph

# Errors of x, y, z in one single graph

# Convert to arrays
t_plot = np.array(t_values, dtype=float)

x_sim = np.array(x_values, dtype=float)
y_sim = np.array(y_values, dtype=float)
z_sim = np.array(z_values, dtype=float)

x_vic = np.array(x_vicon, dtype=float)
y_vic = np.array(y_vicon, dtype=float)
z_vic = np.array(z_vicon, dtype=float)

# Avoid length errors
n = min(len(t_plot), len(x_sim), len(y_sim), len(z_sim),
        len(x_vic), len(y_vic), len(z_vic))

t_plot = t_plot[:n]

x_sim = x_sim[:n]
y_sim = y_sim[:n]
z_sim = z_sim[:n]

x_vic = x_vic[:n]
y_vic = y_vic[:n]
z_vic = z_vic[:n]

# Errors
error_x = x_sim - x_vic
error_y = y_sim - y_vic
error_z = z_sim - z_vic

# Plot
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(t_plot, error_x, "-o", markersize=4, label="error x")
ax.plot(t_plot, error_y, "-o", markersize=4, label="error y")
ax.plot(t_plot, error_z, "-o", markersize=4, label="error z")

ax.axhline(0, linestyle="--", linewidth=1)

ax.set_xlabel("t [s]")
ax.set_ylabel("error [m]")
ax.set_title("Simulation errors with respect to Vicon")
ax.legend()
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.show()

"""
#remove 2 last values from velocities because we only have 9 vicon values
vx_values = vx_values[:9]
vy_values = vy_values[:9]
vz_values = vz_values[:9]
t_values=t_values[:9]
variables = [
    ("vx", vx_values, vx_vicon, "velocidad [m/s]"),
    ("vy", vy_values, vy_vicon, "velocidad [m/s]"),
    ("vz", vz_values, vz_vicon, "velocidad [m/s]"),
]

for name, sim, vicon, ylabel in variables:
    sim = np.array(sim)
    vicon = np.array(vicon)

    error = sim - vicon

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    # Gráfica principal
    ax[0].plot(t_values, sim, "-", linewidth=2, label=f"{name} simulación")
    ax[0].plot(t_values, vicon, "o", markersize=5, label=f"{name} Vicon")

    ax[0].set_ylabel(ylabel)
    ax[0].set_title(f"{name}: simulación vs Vicon")
    ax[0].legend()
    ax[0].grid()

    # Gráfica del error
    ax[1].plot(t_values, error, "-o", markersize=4)
    ax[1].axhline(0, linestyle="--", linewidth=1)

    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel(f"error {name}")
    ax[1].grid()

    plt.tight_layout()
    plt.show()
#Graphics
"""
#Function designed to generate a C3D file readable by a motion capture analysis program in order to observe the trajectory in 3D
def xyz_to_c3d(x, y, z, dt, output_file="balon.c3d", marker_name="BALON"):

    n_frames = len(x)
    freq = round(1.0 / dt)  # Hz
    points = np.zeros((4, 1, n_frames))
    points[0, 0, :] = x
    points[1, 0, :] = y
    points[2, 0, :] = z
    points[3, 0, :] = 0  # residual

    c = ezc3d.c3d()

    # Basic parameters
    c["parameters"]["POINT"]["RATE"]["value"] = [freq]
    c["parameters"]["POINT"]["LABELS"]["value"] = [marker_name]
    c["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]
    c["data"]["points"] = points
    c.write(output_file)
    print(f"Saved: {output_file}  ({n_frames} frames @ {freq} Hz)")
#Convert m to mm
xyz_to_c3d(
    np.array(x_values) * 1e3,
    np.array(y_values) * 1e3,
    np.array(z_values) * 1e3,
    dt=0.004,
    output_file=r"C:\Users\usuario\Desktop\codigo\balon.c3d"
)
