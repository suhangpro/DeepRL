#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *

class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body, gpu=-1):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.tensor(x))
        y = self.fc_head(phi)
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body, gpu=-1):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        if to_numpy:
            return q.cpu().detach().numpy()
        return q

class ActorCriticNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body, gpu=-1):
        super(ActorCriticNet, self).__init__()
        self.fc_actor = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.fc_critic = layer_init(nn.Linear(body.feature_dim, 1))
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.tensor(x))
        pre_prob = self.fc_actor(phi)
        prob = F.softmax(pre_prob, dim=1)
        log_prob = F.log_softmax(pre_prob, dim=1)
        value = self.fc_critic(phi)
        if to_numpy:
            return prob.cpu().detach().numpy()
        return prob, log_prob, value

class ActorCriticLSTM(nn.Module, BaseNet):
    def __init__(self, action_dim, body, use_internal_state=True, gpu=0):
        super(ActorCriticLSTM, self).__init__()
        self.hidden_dim = body.feature_dim    # lstm retains feature size
        self.fc_actor = layer_init(nn.Linear(body.feature_dim + self.hidden_dim, action_dim))
        self.fc_critic = layer_init(nn.Linear(body.feature_dim + self.hidden_dim, 1))
        self.rnn = nn.LSTMCell(body.feature_dim, self.hidden_dim)
        self.body = body
        self.use_internal_state = use_internal_state
        self.is_rnn = True
        self.hx = None  # size not known
        self.cx = None  # size not known
        self.set_gpu(gpu)

    def rnn_reset_state(self, worker_mask):
        if self.hx is None:
            self.hx = self.tensor(np.zeros((len(worker_mask), self.hidden_dim)))
            self.cx = self.tensor(np.zeros((len(worker_mask), self.hidden_dim)))
        else:
            self.hx = self.hx * self.tensor(worker_mask[:, np.newaxis].astype(np.int))
            self.cx = self.cx * self.tensor(worker_mask[:, np.newaxis].astype(np.int))

    def rnn_detach_state(self):
        self.hx = self.hx.detach()
        self.cx = self.cx.detach()

    def forward(self, x, hx=None, cx=None):
        if self.use_internal_state:
            hx, cx = self.hx, self.cx
        phi = self.body(self.tensor(x))
        hx, cx = self.rnn(phi, (hx, cx))
        phi = torch.cat((phi, hx), dim=1)
        pre_prob = self.fc_actor(phi)
        prob = F.softmax(pre_prob, dim=1)
        log_prob = F.log_softmax(pre_prob, dim=1)
        value = self.fc_critic(phi)
        if self.use_internal_state:
            self.hx, self.cx = hx, cx
            return prob, log_prob, value
        else:
            return prob, log_prob, value, hx, cx


class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body, gpu=-1):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        if to_numpy:
            return prob.cpu().detach().numpy()
        return prob

class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body, gpu=-1):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x, to_numpy=False):
        phi = self.body(self.tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        if to_numpy:
            quantiles = quantiles.cpu().detach().numpy()
        return quantiles

class GaussianActorNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body, gpu=-1):
        super(GaussianActorNet, self).__init__()
        self.fc_action = layer_init(nn.Linear(body.feature_dim, action_dim), 3e-3)
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x):
        x = self.tensor(x)
        phi = self.body(x)
        mean = F.tanh(self.fc_action(phi))
        log_std = self.action_log_std.expand_as(mean)
        std = log_std.exp()
        return mean, std, log_std

class GaussianCriticNet(nn.Module, BaseNet):
    def __init__(self, body, gpu=-1):
        super(GaussianCriticNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1), 3e-3)
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x):
        x = self.tensor(x)
        phi = self.body(x)
        value = self.fc_value(phi)
        return value

class DeterministicActorNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body, gpu=-1):
        super(DeterministicActorNet, self).__init__()
        self.fc_action = layer_init(nn.Linear(body.feature_dim, action_dim), 3e-3)
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x, to_numpy=False):
        x = self.tensor(x)
        phi = self.body(x)
        a = F.tanh(self.fc_action(phi))
        if to_numpy:
            a = a.cpu().detach().numpy()
        return a

class DeterministicCriticNet(nn.Module, BaseNet):
    def __init__(self, body, gpu=-1):
        super(DeterministicCriticNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1), 3e-3)
        self.body = body
        self.set_gpu(gpu)

    def predict(self, x, action):
        x = self.tensor(x)
        action = self.tensor(action)
        phi = self.body(x, action)
        value = self.fc_value(phi)
        return value
