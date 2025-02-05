import argparse
import torch
import torch.nn.functional as Func
from torch import nn
import matplotlib.pyplot as plt
from models.model_params import SimulationParams
from training_params.params import training_params, selected_device
from snopt import SNOpt, ODEFuncBase, ODEBlock
from torchdiffeq import odeint_adjoint as odeint

device = selected_device(training_params.device)

# parser = argparse.ArgumentParser('Neural Value Synthesis demo')
# parser.add_argument('--adjoint', action='store_true')
# args = parser.parse_args()
#
# if args.adjoint:
#     from torchdiffeq import odeint_adjoint as odeint
#     print(f"Using the Adjoint method")
# else:
#     from torchdiffeq import odeint


def plot_2d_funcition(xs: torch.Tensor, ys: torch.Tensor, xy_grid, f_mat, func, trace=None, contour=True):
    assert len(xs) == len(ys)
    trace = trace.detach().clone().cpu().squeeze()
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            in_tensor = torch.tensor((x, y)).view(1, 1, 2).float().to(device)
            f_mat[i, j] = func(0, in_tensor).detach().squeeze()

    [X, Y] = xy_grid
    f_mat = f_mat.cpu()
    plt.clf()
    ax = plt.axes()
    if contour:
        ax.contourf(X, Y, f_mat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    else:
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, f_mat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface')
    ax.set_xlabel('Pos')
    ax.set_ylabel('Vel')
    n_plots = trace.shape[1]
    for i in range(n_plots):
        ax.plot(trace[:, i, 0], trace[:, i, 1])
    plt.pause(0.001)


def decomp_x(x, sim_params: SimulationParams):
    return x[:, :, 0:sim_params.nq].clone(), x[:, :, sim_params.nq:].clone()


def decomp_xd(xd, sim_params: SimulationParams):
    return xd[:, :, 0:sim_params.nv].clone(), xd[:, :, sim_params.nv:].clone()


def compose_xxd(x, acc):
    return torch.cat((x, acc), dim=3)


def compose_acc(x, dt):
    ntime, nsim, r, c = x.shape
    v = x[:, :, :, int(c/2):].clone()
    acc = torch.diff(v, dim=0) / dt
    acc_null = torch.zeros_like((acc[0, :, :, :])).view(1, nsim, r, int(c/2))
    return torch.cat((acc, acc_null), dim=0)


class ProjectedDynamicalSystem(nn.Module):
    def __init__(self, ode: ODEBlock, loss, sim_params: SimulationParams, dynamics=None, mode='proj'):
        super(ProjectedDynamicalSystem, self).__init__()
        self.ode = ode
        self.loss_func = loss
        self.sim_params = sim_params
        self.nsim = sim_params.nsim
        self._dynamics = dynamics
        self.step = .15
        self._h = torch.ones((1, 1, sim_params.nv)) * 0.01
        if mode == 'proj':
            self._ctrl = self.project
        else:
            self._ctrl = self.hjb

        if dynamics is None:
            def dynamics(x, xd):
                v = x[:, :, self.sim_params.nq:].view(self.sim_params.nsim, 1, self.sim_params.nv).clone()
                a = xd.clone()
                return torch.cat((v, a), 2)
            self._dynamics = dynamics

    def hjb(self, x):
        q, v = decomp_x(x, self.sim_params)
        xd = torch.cat((v, torch.zeros_like(v)), 2)

        def dvdx(x, value_net):
            with torch.set_grad_enabled(True):
                x = x.detach().requires_grad_(True)
                value = value_net(x).requires_grad_()
                dvdx = torch.autograd.grad(
                    value, x, grad_outputs=torch.ones_like(value), create_graph=True, only_inputs=True
                )[0]
                return dvdx

        # M = self._dynamics._Mfull(q)
        # C = self._dynamics._Cfull(x)
        Vqd = dvdx(x, self.ode)[:, :, self.sim_params.nq:].clone()
        xd_trans = 0.5 * -Vqd * 0.01 * 10#@ torch.linalg.inv(M) - (C@v.mT).mT
        return xd_trans

    def project(self, x):
        q, v = decomp_x(x, self.sim_params)
        xd = torch.cat((v, torch.zeros_like(v)), 2)
        # x_xd = torch.cat((q, v, torch.zeros_like(v)), 2)

        def dvdx(x, value_net):
            with torch.set_grad_enabled(True):
                x = x.detach().requires_grad_(True)
                value = value_net(x).requires_grad_()
                dvdx = torch.autograd.grad(
                    value, x, grad_outputs=torch.ones_like(value), create_graph=True, only_inputs=True
                )[0]
                return dvdx

        Vx = dvdx(x, self.ode)
        norm = ((Vx @ Vx.mT) + 1e-6).sqrt().view(self.nsim, 1, 1)
        unnorm_porj = Func.relu((Vx @ xd.mT) + self.step * self.loss_func(x))
        xd_trans = - (Vx / norm) * unnorm_porj * 5
        return xd_trans[:, :, self.sim_params.nv:].view(self.sim_params.nsim, 1, self.sim_params.nv)

    def dfdt(self, x):
        # TODO: Either the value function is a function of just the actuation space e.g. the cart or it takes into
        # TODO: the main difference is that the normalised projection is changed depending on what is used
        xd = self._ctrl(x)
        return self._dynamics(x, xd)

        # v = x[:, :, self.sim_params.nq:].view(self.sim_params.nsim, 1, self.sim_params.nv).clone()
        # a = xd.clone()
        # return torch.cat((v, a), 2)

    def forward(self, x):
        xd = self.dfdt(x)
        return xd

    @property
    def odes(self): # in case we have multiple odes, collect them in a list
        return [self.ode]

    @property
    def ode_mods(self): # modules of all ode(s)
        return [mod for mod in self.ode.odefunc.modules()]
