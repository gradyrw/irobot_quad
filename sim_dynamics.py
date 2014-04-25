from lwpr import *
import numpy as np

k1 = 1.0
k2 = .5
dt = .05

def dynamics(s, u):
    if (u[0] > 1.0):
        u[0] = 1.0
    elif (u[0] < -1.0):
        u[0] = -1.0
    if (u[1] > 1.0):
        u[1] = 1.0
    elif (u[1] < -1.0):
        u[1] = -1.0
    s[0] += dt*s[2]
    s[1] += dt*s[3]
    s[2] += dt*u[0]
    s[3] += dt*u[1]

model1 = LWPR(4,1)
model2 = LWPR(4,1)

model1.init_D = 10*np.identity(4)
model1.init_alpha = 0*np.ones([4,4])
model1.norm_in = np.array([2.0, 2.0, 2.0, 2.0])

model2.init_D = 10*np.identity(4)
model2.init_alpha = 0*np.ones([4,4])
model2.norm_in = np.array([2.0, 2.0, 2.0, 2.0])

u = np.zeros(2)
state = np.zeros(4)
U = (np.random.rand(20000)).reshape((10000,2))
input = np.zeros(4)
u = np.zeros(2)

A = np.linspace(-1.0, 1.0, 10)
B = np.linspace(-1.0, 1.0, 10)
C = np.linspace(-1.0, 1.0, 10)
D = np.linspace(-1.0, 1.0, 10)

def x_dd(s):
    return s[2]

def y_dd(s):
    return s[3]

for a in A:
    for b in B:
        for c in C:
            for d in D:
                input = np.array([a,b,c,d])
                model1.update(input, np.array([x_dd(input)]))
                model2.update(input, np.array([y_dd(input)]))

def test(N):
    U = np.random.randn(2*N).reshape((N,2))*1.0
    s1 = np.zeros(4)
    s2 = np.zeros(4)
    for i in range(N):
        u = U[i]
        dynamics(s1, u)
        input = np.array([s2[2], s2[3], U[i,0], U[i,1]])
        if (u[0] > 1.0):
            u[0] = 1.0
        elif (u[0] < -1.0):
            u[0] = -1.0
        if (u[1] > 1.0):
            u[1] = 1.0
        elif (u[1] < -1.0):
            u[1] = -1.0
        s2[0] += dt*s2[2]
        s2[1] += dt*s2[3]
        est1 = model1.predict_conf(input)
        s2[2] += dt*(est1[0] + np.random.randn(1)*est1[1])
        est2 = model2.predict_conf(input)
        s2[3] += dt*(est2[0] + np.random.randn(1)*est2[1])
    print s1
    print s2

model1.write_XML("p_xdot.xml")
model2.write_XML("p_ydot.xml")
