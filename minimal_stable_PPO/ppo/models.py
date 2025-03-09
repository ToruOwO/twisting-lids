import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

        # orthogonal init of weights
        # hidden layers scale np.sqrt(2)
        self.init_weights(self.mlp, [np.sqrt(2)] * len(units))

    def forward(self, x):
        return self.mlp(x)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        self.asymmetric = kwargs.pop('asymmetric', False)
        self.separate_value_mlp = kwargs.pop('separate_value_mlp')

        if self.asymmetric and not self.separate_value_mlp:
            self.separate_value_mlp = True
            print("[WARNING] Auto turn on separate value network when using asymmetric obs")
        
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        
        # Decide value network input shape
        if self.asymmetric:
            self.value_input_shape = kwargs.pop('state_input_shape')
            assert self.value_input_shape > 0, "Should have positive input shape"
        else:
            self.value_input_shape = input_shape[0]

        self.fix_sigma = kwargs.get('fix_sigma', True)
        self.actor_units = kwargs.pop('actor_units')
        self.value_units = kwargs.pop('value_units', self.actor_units)

        actor_input_shape = input_shape[0]
        actor_out_size = self.actor_units[-1]
        
        # Build Actor
        self.actor_mlp = MLP(units=self.actor_units, input_size=actor_input_shape)
        self.mu = torch.nn.Linear(actor_out_size, actions_num)

        if self.fix_sigma:
            self.sigma = nn.Parameter(
                torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
            nn.init.constant_(self.sigma, 0)
            
        else:
            self.sigma = torch.nn.Linear(actor_out_size, actions_num)
        
        # Build Value Network
        if self.separate_value_mlp:
            self.value_mlp = MLP(units=self.value_units, input_size=self.value_input_shape)
            value_out_size = self.value_units[-1]
        else:
            value_out_size = actor_out_size # Use the actor head output.

        self.value = torch.nn.Linear(value_out_size, 1)  # Only used when not separated.
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
     

        # policy output layer with scale 0.01
        # value output layer with scale 1
        # torch.nn.init.orthogonal_(self.value.weight, gain=0.01)
        # torch.nn.init.orthogonal_(self.value.weight, gain=1.0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        obs = obs_dict['obs']

        x_act = self.actor_mlp(obs)
        mu = self.mu(x_act)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']

        x_act = self.actor_mlp(obs)
        mu = self.mu(x_act)

        if self.separate_value_mlp:
            if self.asymmetric:
                value_obs = obs_dict['states']
            else:
                value_obs = obs
            x = self.value_mlp(value_obs)
        
        value = self.value(x)

        if self.fix_sigma:
            sigma = self.sigma
        else:
            sigma = self.sigma(x_act)
        return mu, mu * 0 + sigma, value

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        try:
            distr = torch.distributions.Normal(mu, sigma)
        except ValueError:
            print("NanError mu", mu)
            print("NanError sigma", sigma)
            raise ValueError
        
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
        }
        return result
