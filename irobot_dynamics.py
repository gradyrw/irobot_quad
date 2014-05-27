from lwpr import *
import numpy as np

w = .258
k1 = 1.0
k2 = 1.0
dt = .02

def dynamics(s, u):
    s[0] += dt*(s[3] + s[4])/2.0*np.cos(s[2])
    s[1] += dt*(s[3] + s[4])/2.0*np.sin(s[2])
    s[2] += dt*(s[4] - s[3])/w
    s[3] += dt*k1*(u[0] - s[3])
    s[4] += dt*k2*(u[1] - s[4])

model1 = LWPR(5,1)
model2 = LWPR(5,1)
model3 = LWPR(5,1)

model1.init_D = 10*np.identity(5)
#model1.init_alpha = 0*np.ones([5,5])
model1.norm_in = np.array([20.0, 20.0, 6.28, 4.0, 4.0])

model2.init_D = 10*np.identity(5)
#model2.init_alpha = 0*np.ones([5,5])
model2.norm_in = np.array([20.0, 20.0, 6.28, 4.0, 4.0])

model3.init_D = 10*np.identity(5)
#model3.init_alpha = 0*np.ones([5,5])
model3.norm_in = np.array([20.0, 20.0, 6.28, 4.0, 4.0])

A = np.linspace(-2.0, 2.0, 10)
B = np.linspace(-2.0, 2.0, 10)
C = np.linspace(-2.0, 2.0, 10)
D = np.linspace(-2.0, 2.0, 10)
E = np.linspace(-2.0, 2.0, 10)

def px_d(s):
    return (s[3] + s[4])/2.0*np.cos(s[2])

def py_d(s):
    return (s[3] + s[4])/2.0*np.sin(s[2])

def theta_d(s):
    return (s[4] - s[3])/w

for a in A:
    for b in B:
        for c in C:
            for d in D:
                for e in E:
                    input = np.array([a,b,c,d,e])
                    model1.update(input, np.array([px_d(input)]))
                    model2.update(input, np.array([py_d(input)]))
                    model3.update(input, np.array([theta_d(input)]))


u1 = np.random.randn(1000*2)*.33 + 1.0
u2 = np.random.randn(1000*2)*.33 - 1.0
u3 = np.random.randn(1000*2)*.33
for i in range(1000):
    u3[2*i] += 1.0
    u3[2*i + 1] += 0
u4 = np.random.randn(1000*2)
for i in range(1000):
    u3[2*i] += 0
    u3[2*i + 1] += 1.0
u5 = np.random.randn(1000*2)
U = [u1,u2,u3,u4,u5]

def test1(N):
    U = np.random.randn(2*N).reshape((N,2))*.25 + 1.0
    s1 = np.zeros(5)
    s2 = np.zeros(5)
    for i in range(N):
        u = U[i]
        dynamics(s1, u)
        input = s2
        s2[0] += dt*model1.predict(input)
        s2[1] += dt*model2.predict(input)
        s2[2] += dt*model3.predict(input)
        s2[3] += dt*k1*(u[0] - s2[3])
        s2[4] += dt*k2*(u[1] - s2[4])
    print s1
    print s2

def test2(N):
    U = np.random.randn(2*N).reshape((N,2))*.25 - 1.0
    s1 = np.zeros(5)
    s2 = np.zeros(5)
    for i in range(N):
        u = U[i]
        dynamics(s1, u)
        input = s2
        s2[0] += dt*model1.predict(input)
        s2[1] += dt*model2.predict(input)
        s2[2] += dt*model3.predict(input)
        s2[3] += dt*k1*(u[0] - s2[3])
        s2[4] += dt*k2*(u[1] - s2[4])
    print s1
    print s2

def test3(N):
    U = np.random.randn(2*N).reshape((N,2))*1.0
    s1 = np.zeros(5)
    s2 = np.zeros(5)
    for i in range(N):
        u = U[i]
        dynamics(s1, u)
        input = s2
        s2[0] += dt*model1.predict(input)
        s2[1] += dt*model2.predict(input)
        s2[2] += dt*model3.predict(input)
        s2[3] += dt*k1*(u[0] - s2[3])
        s2[4] += dt*k2*(u[1] - s2[4])
    print s1
    print s2


def epoch():
    model1.init_D = 50.0*np.identity(5)
    model2.init_D = 50.0*np.identity(5)
    model3.init_D = 50.0*np.identity(5)
    for u in U:
        state = np.zeros(5)
        for i in range(1000):
            input = np.copy(state)
            dynamics(state, u[2*i:2*(i+1)])
            model1.update(input, np.array([px_d(input)]))
            model2.update(input, np.array([py_d(input)]))
            model3.update(input, np.array([theta_d(input)]))

for i in range(10):
    epoch()

model1.write_XML("x.xml")
model2.write_XML("y.xml")
model3.write_XML("theta.xml")
        


