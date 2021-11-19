'''
problem setup
u(x, y) = 1 - exp(lam*x) * cos(2*pi*y)
v(x, y) = lam / (2*pi) * exp(lam*x) * sin(2*pi*y)
p(x, y) = 1/2 * (1-exp(2*lam*x))
lam = 1/(2*mu) - sqrt(1/(4*mu**2) + 4*pi**2) mu=1/Re=1/40
for
x in [-0.5, 1.0]
y in [-0.5, 1.5]


inputs: [x, y]
outputs: [u, v, p], [u, v] is velocity in [x, y]
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt

gpu = 4
torch.cuda.set_device(gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VP_NSFnet(object):
    def __init__(self, xb, yb, ub, vb, x, y, layers) -> None:
        self.update_data(xb, yb, x, y)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        self.vb = torch.tensor(vb, dtype=torch.float32).to(device)

        self.null =  torch.zeros(*x.shape).to(device) # zeros as labels for (x, y)
        self.loss_func = nn.MSELoss()

        self.build_model(layers)

        # weights for loss boundary
        self.alpha = 1.0
        self.adaptive_alpha = False

        self.iter = 0
        self.loss_list = []
        self.loss_b_list = []
        self.loss_f_list = []


    def update_data(self, xb, yb, x, y) -> None:
        # bourndary points
        self.x_b = torch.tensor(xb.reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(device)
        self.y_b = torch.tensor(yb.reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(device)

        # collocation points
        self.x_f = torch.tensor(x.reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(device)
        self.y_f = torch.tensor(y.reshape(-1, 1),
                                dtype=torch.float32,
                                requires_grad=True).to(device)

    def build_model(self, layers):
        nnlist = []
        for i in range(len(layers) - 2):
            nnlist.append(nn.Linear(layers[i], layers[i+1]))
            nnlist.append(nn.Tanh())
        nnlist.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*nnlist).to(device)
        # print(self.model)

    def net_NS(self, x, y):
        output = self.net(torch.hstack((x, y)))
        u, v, p = torch.hsplit(output, output.size(1))
        return u, v, p

    def net_f_NS(self, x, y):
        u, v, p = self.net_NS(x, y)
        # options
        # grad_outputs: 指定输出形状？
        # retain_graph: 如果为false, 那么计算梯度后计算图会释放掉，如果为None，默认为create_graph
        # create_graph: 是否创建计算图，为True可以继续计算高阶导数
        # options = {
        #     'grad_outputs': torch.ones_like(u),
        #     'retain_graph': True,
        #     'create_graph': True
        # }
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]

        v_x = torch.autograd.grad(
            v, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        v_y = torch.autograd.grad(
            v, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        v_xx = torch.autograd.grad(
            v_x, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        v_yy = torch.autograd.grad(
            v_y, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]

        p_x = torch.autograd.grad(
            p, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        p_y = torch.autograd.grad(
            p, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True)[0]
        
        f_e1 = (u * u_x + v * u_y) + p_x - (1.0/40) * (u_xx + u_yy)
        f_e2 = (u * v_x + v * v_y) + p_y - (1.0/40) * (v_xx + v_yy)
        f_e4 = u_x + v_y
        f_e = f_e1 + f_e2 + f_e4

        return f_e

    def update_alpha(self):
        pass

    def Adam_train(self, max_iter=10000, lr=1e-3):
        adam_optimizer = optim.Adam(self.net.parameters(),
                                    lr=lr,
                                    weight_decay=1e-4)
        adam_scheduler = lr_scheduler.StepLR(adam_optimizer,
                                             step_size=10000,
                                             gamma=0.1)

        for it in range(max_iter):
            self.net.train()
            adam_optimizer.zero_grad()
            # boundary data & f data
            ub_pred, vb_pred, pb_pred = self.net_NS(self.x_b, self.y_b)
            f_pred = self.net_f_NS(self.x_f, self.y_f)

            loss_b = self.loss_func(ub_pred, self.ub) + self.loss_func(vb_pred, self.vb)
            loss_f = self.loss_func(f_pred, self.null)
            loss = loss_f + self.alpha * loss_b

            loss.backward()

            adam_optimizer.step()
            adam_scheduler.step()

            self.iter += 1
            self.loss_list.append(loss.cpu().item())
            self.loss_b_list.append(loss_b.cpu().item())
            self.loss_f_list.append(loss_f.cpu().item())
            if self.iter % 1000 == 0:
                print(self.iter, loss.cpu().item(), (loss_b / loss_f).cpu().item())

    def closure(self):
        self.lbfgs_optimizer.zero_grad()
        self.net.train()

        ub_pred, vb_pred, pb_pred = self.net_NS(self.x_b, self.y_b)
        f_pred = self.net_f_NS(self.x_f, self.y_f)

        loss_b = self.loss_func(ub_pred, self.ub) + self.loss_func(vb_pred, self.vb)
        loss_f = self.loss_func(f_pred, self.null)
        loss = loss_f + self.alpha * loss_b

        loss.backward()

        self.iter += 1
        self.loss_list.append(loss.cpu().item())
        self.loss_b_list.append(loss_b.cpu().item())
        self.loss_f_list.append(loss_f.cpu().item())
        if self.iter % 100 == 0:
            print(self.iter, loss.cpu().item(), (loss_b / loss_f).cpu().item())

        return loss

    def BFGS_train(self, max_iter=10000, lr=1):
        self.lbfgs_optimizer = torch.optim.LBFGS(self.net.parameters(),
                                                lr=lr,
                                                max_iter=max_iter,
                                                # max_eval=int(iters*1.25),
                                                # # max_eval=50000,
                                                # history_size=100,
                                                tolerance_grad=1e-9,
                                                tolerance_change=0.5 * np.finfo(float).eps,
                                                line_search_fn="strong_wolfe"
                                                )

        self.lbfgs_optimizer.step(self.closure)

    def predict(self):
        pass

    def plot(self):
        def f(X, Y):
            shape = X.shape
            x = torch.tensor(X, dtype=torch.float32).reshape(-1, 1)
            y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)
            u, v, p = self.net_NS(x.to(device), y.to(device))
            u_field = u.detach().cpu().numpy().reshape(*shape)
            v_field = v.detach().cpu().numpy().reshape(*shape)
            p_field = p.detach().cpu().numpy().reshape(*shape)
            return u_field, v_field, p_field

        # loss line
        plt.figure()
        plt.title('loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # plt.plot(self.loss_list)
        plt.plot(self.loss_b_list, label='Loss_b')
        plt.plot(self.loss_f_list, label='Loss_f')
        plt.legend()
        plt.savefig('img/loss.png')

        x = np.linspace(-0.5, 1, 150)
        y = np.linspace(-0.5, 1.5, 200)
        X, Y = np.meshgrid(x, y)
        u_field, v_field, p_field = f(X, Y)

        # velocity field
        plt.figure(figsize=(6, 8))
        plt.title('velocity')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim((-0.5, 1.0))
        plt.ylim((-0.5, 1.5))
        plt.streamplot(x, y, u_field, v_field)
        plt.savefig('img/velocity_steam.png')

        # pressure field
        plt.figure(figsize=(6, 8))
        plt.title('pressure.png')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim((-0.5, 1.0))
        plt.ylim((-0.5, 1.5))
        plt.contourf(X, Y, p_field)
        plt.colorbar()
        plt.savefig('img/pressure_field.png')

def ground_truth(x, y):
    u = 1 - np.exp(lam * x) * np.cos(2 * np.pi * y) # velocity in x and y direction
    v = lam / (2 * np.pi) * np.exp(lam * x) * np.sin(2 * np.pi * y)
    p = 0.5 * (1 - np.exp(2*lam*x))
    return u, v, p

if __name__ == "__main__":

    N_train = 1000
    N_bc = 50
    layers = [2, 50, 50, 50, 50, 3]

    # Load Data
    Re = 40
    lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

    x_bc_upper = np.random.rand(N_bc, 1) * 1.5 - 0.5
    y_bc_upper = np.ones((N_bc, 1)) * 1.5

    x_bc_lower = np.random.rand(N_bc, 1) * 1.5 - 0.5
    y_bc_lower = np.ones((N_bc, 1)) * (-0.5)

    x_bc_left = np.ones((N_bc, 1)) * (-0.5)
    y_bc_left = np.random.rand(N_bc, 1) * 2.0 - 0.5

    x_bc_right = np.ones((N_bc, 1)) * 1.0
    y_bc_right = np.random.rand(N_bc, 1) * 2.0 - 0.5

    # boundary data
    xb_train = np.vstack([x_bc_upper, x_bc_lower, x_bc_left, x_bc_right])
    yb_train = np.vstack([y_bc_upper, y_bc_lower, y_bc_left, y_bc_right])
    Xb_train = np.hstack([xb_train, yb_train])
    # ub_train = 1 - np.exp(lam * xb_train) * np.cos(2 * np.pi * yb_train) # velocity in x and y direction
    # vb_train = lam / (2 * np.pi) * np.exp(lam * xb_train) * np.sin(2 * np.pi * yb_train)
    ub_train, vb_train, _ = ground_truth(xb_train, yb_train)

    # function loss data
    xf_train = np.random.rand(N_train, 1) * 1.5 - 0.5
    yf_train = np.random.rand(N_train, 1) * 2.0 - 0.5
    xf_train = np.vstack([xf_train, xb_train]) # 加上边界点
    yf_train = np.vstack([yf_train, yb_train])

    model = VP_NSFnet(xb_train, yb_train, ub_train, vb_train, xf_train, yf_train, layers)
    model.Adam_train(max_iter=30000, lr=1e-3)
    model.BFGS_train(max_iter=10000, lr=1)
    model.plot()
    print(len(model.loss_list))
