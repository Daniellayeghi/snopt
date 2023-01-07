import random
import util, options
from torchdiffeq import odeint_adjoint as odesolve
from models.cartpole_animation import animate_cartpole
from models.cartpole import Cartpole
from models.model_params import ModelParams, SimulationParams
from training_params.params import selected_device, training_params
from neural_value_synthesiser import *
import matplotlib.pyplot as plt

device = selected_device(training_params.device)

cp_params = ModelParams(2, 2, 1, 4, 4)
sim_params = SimulationParams(cp_params, 6, 4, 2, 2, 2, 1, 300)

prev_cost, diff, iteration, tol, max_iter, step_size = 0, 100.0, 1, 0, 3000, 1.0
Q = torch.diag(torch.Tensor([1, 2, 2, 0.01, 0.01])).repeat(sim_params.nsim, 1, 1).to(device)
R = torch.diag(torch.Tensor([0.1])).repeat(sim_params.nsim, 1, 1).to(device)
Qf = torch.diag(torch.Tensor([50000, 600000, 600000, 500, 6000])).repeat(sim_params.nsim, 1, 1).to(device)



def wrap_free_state(x: torch.Tensor):
    q, v = x[:, :, :sim_params.nq], x[:, :, sim_params.nq:]
    q_new = torch.cat((torch.sin(q[:, :, 1]), torch.cos(q[:, :, 1]) - 1, q[:, :, 0]), 1).unsqueeze(1)
    return torch.cat((q_new, v), 2)


def wrap_free_state_batch(x: torch.Tensor):
    q, v = x[:, :, :, :sim_params.nq], x[:, :, :, sim_params.nq:]
    q_new = torch.cat((torch.sin(q[:, :, :, 1]), torch.cos(q[:, :, :, 1]) - 1, q[:, :, :, 0]), 2).unsqueeze(2)
    return torch.cat((q_new, v), 3)


def bounded_state(x: torch.Tensor):
    qc, qp, qdc, qdp  = x[:, :, 0].clone().unsqueeze(1), x[:, :, 1].clone().unsqueeze(1), x[:, :, 2].clone().unsqueeze(1), x[:, :, 3].clone().unsqueeze(1)
    qp = (qp+2 * torch.pi)%torch.pi
    return torch.cat((qc, qp, qdc, qdp), 2)


def bounded_traj(x: torch.Tensor):
    def bound(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    qc, qp, qdc, qdp = x[:, :, :, 0].clone().unsqueeze(2), x[:, :, :, 1].clone().unsqueeze(2), x[:, :, :, 2].clone().unsqueeze(2), x[:, :, :, 3].clone().unsqueeze(2)
    qp = bound(qp)
    return torch.cat((qc, qp, qdc, qdp), 3)


class NNValueFunction(ODEFuncBase):
    def __init__(self, n_in, opt):
        super(NNValueFunction, self).__init__(opt)

        self.nn = nn.Sequential(
            nn.Linear(n_in, 32, bias=False),
            nn.Softplus(),
            nn.Linear(32, 64, bias=False),
            nn.Softplus(),
            nn.Linear(64, 256, bias=False),
            nn.Softplus(),
            nn.Linear(256, 64, bias=False),
            nn.Softplus(),
            nn.Linear(64, 32, bias=False),
            nn.Softplus(),
            nn.Linear(32, 1, bias=False)
        )

        def init_weights(net):
            if type(net) == nn.Linear:
                torch.nn.init.xavier_uniform(net.weight)

        self.nn.apply(init_weights)

    def F(self, t, x):
        return self.nn(x)


bad_init = False




bad_init = False

if __name__ == "__main__":

    opt = options.set()
    training_params.device = opt.gpu
    training_params.nsims = opt.batch_size
    training_params.nepochs = opt.epoch
    sim_params.nsim = training_params.nsims

    print("Training")
    print(f"Running Experiment With nsims: {training_params.nsims}, neposchs {training_params.nepochs}")

    cartpole = Cartpole(sim_params.nsim, cp_params)
    # nn_value_func = NNValueFunction(sim_params.nqv).to(device)
    # dyn_system = ProjectedDynamicalSystem(nn_value_func, loss_func, sim_params, cartpole, mode='proj').to(device)
    # time = torch.linspace(0, (sim_params.ntime - 1) * 0.01, sim_params.ntime).to(device)

    def loss_func(x: torch.Tensor):
        x = wrap_free_state(x)
        return x @ Q @ x.mT


    def batch_state_loss(x: torch.Tensor):
        x = wrap_free_state_batch(x)
        t, nsim, r, c = x.shape
        x_run = x[0:-1, :, :, :].view(t - 1, nsim, r, c).clone()
        x_final = x[-1:, :, :, :].view(1, nsim, r, c).clone()
        l_running = torch.sum(x_run @ Q @ x_run.mT, 0).squeeze()
        l_terminal = (x_final @ Qf @ x_final.mT).squeeze()

        return torch.mean(l_running + l_terminal)


    def batch_ctrl_loss(acc: torch.Tensor):
        qddc = acc[:, :, :, 0].unsqueeze(2).clone()
        l_ctrl = torch.sum(qddc @ R @ qddc.mT, 0).squeeze()
        return torch.mean(l_ctrl)


    def loss_function(x, acc):
        return batch_ctrl_loss(acc) + batch_state_loss(x)


    odefunc = NNValueFunction(sim_params.nqv, opt).to(device)
    integration_time = torch.tensor([0.0, 2.0]).float()
    ode = ODEBlock(opt, odefunc, odesolve, integration_time)
    net = ProjectedDynamicalSystem(ode,  loss_func, sim_params, cartpole, mode='proj').to(device)

    precond = SNOpt(net, eps=0.05, update_freq=100)
    optim = torch.optim.Adam(net.parameters(), lr=0.001)

    # q_init = torch.Tensor([0, torch.pi]).repeat(sim_params.nsim, 1, 1).to(device)
    qp_init = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(0, 1.9 * 3.14) * 1
    qc_init = torch.FloatTensor(sim_params.nsim, 1, 1).uniform_(-2, 2) * 1
    qd_init = torch.FloatTensor(sim_params.nsim, 1, sim_params.nv).uniform_(0, 0) * 1
    x_init = torch.cat((qc_init, qp_init, qd_init), 2).to(device)

    while iteration < training_params.nepochs:
        # if bad_init:
        #     nn_value_func = NNValueFunction(sim_params.nqv).to(device)
        #     dyn_system = ProjectedDynamicalSystem(nn_value_func, loss_func, sim_params, cartpole).to(device)

        precond.train_itr_setup()  # <--- additional step for precond
        optim.zero_grad()
        traj = net(x_init)
        acc = compose_acc(traj, 0.02)
        xxd = compose_xxd(traj, acc)
        loss = loss_function(traj, acc)
        loss.backward()
        precond.step()  # <--- additional step for precond
        optim.step()
        # for param in dyn_system.parameters():
        #     print(f"\n{param}\n")

        print(f"Epochs: {iteration}, Loss: {loss.item()}, iteration: {iteration % 10}")

        selection = random.randint(0, sim_params.nsim - 1)

        # if iteration % 1 == 0:
        #     fig_1 = plt.figure(1)
        #     for i in range(sim_params.nsim):
        #         # traj_b = bounded_traj(traj)
        #         qpole = traj[:, i, 0, 1].clone().detach()
        #         qdpole = traj[:, i, 0, 3].clone().detach()
        #         plt.plot(qpole, qdpole)
        #
        #     plt.pause(0.001)
        #     fig_2 = plt.figure(2)
        #     ax_2 = plt.axes()
        #     plt.plot(traj[:, selection, 0, 0].clone().detach())
        #     plt.plot(qpole)
        #     plt.plot(acc[:, selection, 0, 0].clone().detach())
        #     plt.plot(acc[:, selection, 0, 1].clone().detach())
        #     plt.pause(0.001)
        #     ax_2.set_title(loss.item)
        #     fig_1.clf()
        #     fig_2.clf()
        #
        # if iteration % 10 == 0:
        #     cart = traj[:, selection, 0, 0].clone().detach().numpy()
        #     pole = traj[:, selection, 0, 1].clone().detach().numpy()
        #     animate_cartpole(cart, pole)

        iteration += 1
