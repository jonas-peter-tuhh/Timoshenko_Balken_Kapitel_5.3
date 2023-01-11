# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:20:05 2022
@author: Jonas Peter
"""
##Wichtig!
##Wichtig!
##!!!Achtung!!! Helpers.py Datei notwendig!!! siehe Helpers-respository --> |https://github.com/jonas-peter-tuhh/Helpers|
#Helpers.py - Datei muss einen Ordner über entsprechenden PINN abgespeichert werden!
import sys
import os
#insert path of parent folder to import helpers
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from helpers import *
import warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train = True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 70)
        self.output_layer = nn.Linear(70, 1)

    def forward(self, x):  # ,p,px):
        inputs = torch.unsqueeze(x, 1)
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        output = self.output_layer(layer1_out)
        return output

net_B = Net()
net_S = Net()
net_B = net_B.to(device)
net_S = net_S.to(device)
##
choice_load = input("Möchtest du ein State_Dict laden? (y/n): ")
if choice_load == 'y':
    train = False
    filename = input("Welches State_Dict möchtest du laden? ")
    path = os.path.join('saved_data', filename)
    net_S.load_state_dict(torch.load(path + '_S'))
    net_B.load_state_dict(torch.load(path + '_B'))
    net_S.eval()
    net_B.eval()
##
# Hyperparameter
learning_rate = 0.01
mse_cost_function = torch.nn.MSELoss()  # Mean squared error
optimizer = torch.optim.Adam([{'params': net_B.parameters()}, {'params': net_S.parameters()}], lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, verbose=True, factor=0.75)


# Definition der Parameter des statischen Ersatzsystems
Lb = float(input('Länge des Kragarms [m]: '))
EI = 21
K = 5/6
G = 80
A = 100

#Normierungsfaktor (siehe Kapitel 10.3)
normfactor = 10/((11*Lb**5)/(120*EI))

# ODE als Loss-Funktion, Streckenlast
Ln = 0 #float(input('Länge Einspannung bis Anfang der ' + str(i + 1) + '. Streckenlast [m]: '))
Lq = Lb # float(input('Länge der ' + str(i + 1) + '. Streckenlast [m]: '))
s = str(normfactor)+"*x"#input(str(i + 1) + '. Streckenlast eingeben: ')


def h(x):
    return eval(s)

#Netzwerk für Biegung
def f(x, net_B):
    u = net_B(x)
    _, _, _, u_xxxx = deriv(u, x, 4)
    ode = u_xxxx + (h(x - Ln))/EI
    return ode

#Netzwerk für Schub
def g(x, net_S):
    u = net_S(x)
    _, u_xx = deriv(u, x, 2)
    ode = u_xx - (h(x - Ln))/ (K * A * G)
    return ode


x = np.linspace(0, Lb, 1000)
pt_x = myconverter(x)
qx = h(x)* (x <= (Ln + Lq)) * (x >= Ln)


Q0 = integrate.cumtrapz(qx, x, initial=0)

qxx = (qx) * x

M0 = integrate.cumtrapz(qxx, x, initial=0)
if train:
    y1 = net_B(torch.unsqueeze(myconverter(x, False),1)) + net_S(torch.unsqueeze(myconverter(x, False),1))
    fig = plt.figure()
    plt.grid()
    ax1 = fig.add_subplot()
    ax1.set_xlim([0, Lb])
    ax1.set_ylim([-10, 0])
    net_out_plot = myconverter(y1)
    line1, = ax1.plot(x, net_out_plot)
    plt.show(block = False)
    pt_x = torch.unsqueeze(myconverter(x),1)
    f_anal = (-1 / 120 * normfactor * x ** 5 + 1 / 6 * Q0[-1] * x ** 3 - M0[-1] / 2 * x ** 2) / EI + (
            1 / 6 * normfactor * x ** 3 - Q0[-1] * x) / (K * A * G)
##
iterations = 100000
for epoch in range(iterations):
    if not train: break
    optimizer.zero_grad()  # to make the gradients zero
    x_bc = np.linspace(0, Lb, 500)
    pt_x_bc = myconverter(x_bc)

    x_collocation = np.random.uniform(low=0.0, high=Lb, size=(250 * int(Lb), 1))
    all_zeros = np.zeros((250 * int(Lb), 1))

    pt_x_collocation = myconverter(x_collocation)
    pt_all_zeros = myconverter(all_zeros,False)
    f_out_B = f(pt_x_collocation, net_B)
    f_out_S = g(pt_x_collocation, net_S)

    # Randbedingungen
    net_bc_out_B = net_B(pt_x_bc)
    net_bc_out_S = net_S(pt_x_bc)
    vb_x, vb_xx, vb_xxx = deriv(net_bc_out_B, pt_x_bc, 3)
    vs_x = deriv(net_bc_out_S, pt_x_bc, 1)

    #RB für Biegung
    BC3  = net_bc_out_B[0]
    BC6  = vb_xxx[0] - Q0[-1] / EI
    BC7  = vb_xxx[-1]
    BC8  = vb_xx[0] + M0[-1] / EI
    BC9  = vb_xx[-1]
    BC10 = vb_x[0]


    #RB für Schub
    BC2 = net_bc_out_S[0]
    BC4 = vs_x[0] + Q0[-1]/(K*A*G)
    BC5 = vs_x[-1]

    mse_Gamma_B = errsum(mse_cost_function, BC3, 1 / normfactor * BC6, BC7, 1 / normfactor * BC8, BC9, BC10)
    mse_Gamma_S = errsum(mse_cost_function, BC2, 1 / normfactor * BC4, BC5)
    mse_Omega_B = errsum(mse_cost_function, f_out_B)
    mse_Omega_S = errsum(mse_cost_function, f_out_S)

    loss = mse_Gamma_B + mse_Gamma_S + mse_Omega_B + mse_Omega_S

    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    with torch.autograd.no_grad():
        if epoch % 10 == 9:
            print(epoch, "Training Loss:", loss.data)
            plt.grid()
            net_out = myconverter(net_B(pt_x) + net_S(pt_x))
            err = np.linalg.norm(net_out - f_anal, 2)
            print(f'Error = {err}')
            if err < 0.1*Lb:
                print(f"Die L^2 Norm des Fehlers ist {err}.\nStoppe Lernprozess")
                break
            line1.set_ydata(net_B(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device),1)).cpu().detach().numpy() + net_S(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device),1)).cpu().detach().numpy())
            fig.canvas.draw()
            fig.canvas.flush_events()
##
if choice_load == 'n':
    choice_save = input("Möchtest du die Netzwerkparameter abspeichern? (y/n): ")
    if choice_save == 'y':
        filename = input("Wie soll das State_Dict heißen?")
        path = os.path.join('saved_data', filename)
        torch.save(net_S.state_dict(), path + '_S')
        torch.save(net_B.state_dict(), path + '_B')

##
pt_x = torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=True).to(device), 1)

pt_u_out_s = net_S(pt_x)
pt_u_out = net_B(pt_x)
w_x = torch.autograd.grad(pt_u_out, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(pt_u_out))[0]
w_xx = torch.autograd.grad(w_x, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(pt_u_out))[0]
w_xxx = torch.autograd.grad(w_xx, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(pt_u_out))[0]
w_xxxx = torch.autograd.grad(w_xxx, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(pt_u_out))[0]

ws_x = torch.autograd.grad(pt_u_out_s, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(pt_u_out))[0]
ws_xx = torch.autograd.grad(ws_x, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(pt_u_out))[0]

w_x = w_x.cpu().detach().numpy()
w_xx = w_xx.cpu().detach().numpy()
w_xxx = w_xxx.cpu().detach().numpy()
w_xxxx = w_xxxx.cpu().detach().numpy()

ws_x = ws_x.cpu().detach().numpy()
ws_xx = ws_xx.cpu().detach().numpy()

#Biegung Kompatibilität Numpy Array
v_out_cpu = pt_u_out.cpu()
v_out = v_out_cpu.detach()
v_out = v_out.numpy()

#Schub Kompatibilität Numpy Array
s_out_cpu = pt_u_out_s.cpu()
s_out = s_out_cpu.detach()
s_out = s_out.numpy()

fig = plt.figure()

#Euler-Bernoulli Plots
#Die Funktionen, die dort stehen sind Relikte von früheren Testläufen (analytische Lösungen)
plt.subplot(2, 2, 1)
plt.title('$v_{b}$ Auslenkung (aus Biegung)')
plt.xlabel('')
plt.ylabel('[cm]')
plt.plot(x, v_out)
plt.plot(x, (-1/120 * normfactor * x**5 + Q0[-1]/6 * x**3 - M0[-1]/2 * x**2)/EI)
plt.grid()

plt.subplot(2, 2, 3)
plt.title('$\phi$ Neigung')
plt.xlabel('')
plt.ylabel('$10^{-2}$')
plt.plot(x, w_x)
plt.plot(x, (-1/24 * normfactor * x**4 + 0.5 * Q0[-1] * x**2 - M0[-1] * x)/EI)
plt.grid()



#Timoshenko Plots

plt.subplot(2, 2, 2)
plt.title('$v_{s}$ Auslenkung (aus Schub)')
plt.xlabel('')
plt.ylabel('$cm$')
plt.plot(x, s_out)
plt.plot(x, (1/6 * normfactor * x**3 - Q0[-1] * x)/(K*A*G))
plt.grid()

plt.subplot(2, 2, 4)
plt.title('Schubwinkel $\gamma$')
plt.xlabel('')
plt.ylabel('$(10^{-2})$')
plt.plot(x, ws_x)
plt.plot(x, (normfactor * 0.5 * x**2 - Q0[-1])/(K*A*G))
plt.grid()

phi_anal = (-1/24 * normfactor * x**4 + 0.5 * Q0[-1] * x**2 - M0[-1] * x)/EI
phi_net = w_x
phi_err = np.linalg.norm((phi_net-phi_anal), 2)/np.linalg.norm(phi_anal, 2)
print('phi_err=',phi_err)

gamma_anal = ((normfactor * 0.5 * x**2 - Q0[-1])/(K*A*G))
gamma_net = ws_x
gamma_err = np.linalg.norm((gamma_net-gamma_anal), 2)/np.linalg.norm(gamma_anal, 2)
print('\u03B3 5.1 =',gamma_err)

vs_anal = (1/6 * normfactor * x**3 - Q0[-1] * x)/(K*A*G)
vs_net = s_out
vs_err = np.linalg.norm((vs_net-vs_anal), 2)/np.linalg.norm(vs_anal, 2)
print('vs_err=',vs_err)
plt.show()
##
plt.plot()
plt.title('Schubwinkel $\gamma$')
plt.xlabel('')
plt.ylabel('$(10^{-2})$')
plt.plot(x, (ws_x))
plt.plot(x, ((normfactor * 0.5 * x**2 - Q0[-1])/(K*A*G)))
plt.legend(['$\gamma_{out}$','$\gamma_{anal}$'])
plt.grid()
##
plt.plot()
plt.title('Verschiebung der neutralen Faser durch Schub $v_{s}$')
plt.xlabel('')
plt.ylabel('$[cm]$')
plt.plot(x, (s_out))
plt.plot(x, (1/6 * normfactor * x**3 - Q0[-1] * x)/(K*A*G))
plt.legend(['$v_{s,out}$','$v_{s,anal}$'])
plt.grid()
##

