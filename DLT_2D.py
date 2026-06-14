import numpy as np

#Test measures (x,y coordinates of each point)
P1=(0,0) #Down left RASI 
P2=(0,922) #Up left LASI 
P3=(1147,922) #Up right LPSI
P4=(1147,0) #Down right RPSI 

#Coordinates seen on camera
U1=-164.31 #Px
V1=-9.24
U2=-160.41
V2=269.62
U3=142.83
V3=230.62
U4=136.98
V4=-17.04

#A*L = B
A=  [[P1[0],P1[1],1,-U1*P1[0],-U1*P1[1],0,0,0],
    [P2[0],P2[1],1,-U2*P2[0],-U2*P2[1],0,0,0],
    [P3[0],P3[1],1,-U3*P3[0],-U3*P3[1],0,0,0],
    [P4[0],P4[1],1,-U4*P4[0],-U4*P4[1],0,0,0],
    [0,0,0,-V1*P1[0],-V1*P1[1],P1[0],P1[1],1],
    [0,0,0,-V2*P2[0],-V2*P2[1],P2[0],P2[1],1],
    [0,0,0,-V3*P3[0],-V3*P3[1],P3[0],P3[1],1],
    [0,0,0,-V4*P4[0],-V4*P4[1],P4[0],P4[1],1]
    ]

B=np.array([U1,U2,U3,U4,V1,V2,V3,V4])
L=np.linalg.solve(A,B)
print("The DLT 2D Parameters are: ",L)
U5=float(input("U5= "))
V5=float(input("V5= "))
C= np.array([
    [U5*L[3] - L[0], U5*L[4] - L[1]],
    [V5*L[3] - L[5], V5*L[4] - L[6]]
])

D = np.array([
    L[2] - U5,
    L[7] - V5
])

X5, Y5 = np.linalg.solve(C, D)
print("X= ",X5," mm ;","Y= ",Y5," mm")