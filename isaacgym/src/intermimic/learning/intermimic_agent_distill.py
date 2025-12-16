# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from isaacgym.torch_utils import *

import numpy as np
import torch 
from torch import nn

from . import intermimic_agent


class InterMimicAgentDistill(intermimic_agent.InterMimicAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        self.expert_loss_coef = config['expert_loss_coef']
        self.entropy_coef = config['entropy_coef']
        self.ev_ma            = 0.0   # running avg explained‑variance
        self.critic_win_streak = 0    # consecutive windows EV ≥ threshold
        self.actor_update_num = 0
        return


    def init_tensors(self):
        super().init_tensors()
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['expert_mask'] = torch.zeros(batch_shape, dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['expert'] = torch.zeros((*batch_shape, 153), dtype=torch.float32, device=self.ppo_device)
        self.tensor_list += ['amp_obs', 'rand_action_mask', 'expert', 'expert_mask']
        return


    def play_steps(self):
        self.set_eval()

        epinfos = []
        update_list = self.update_list

        # Initialize DAgger beta coefficient
        beta_t = max(1 - max((self.epoch_num - 500) / 5000, 0), 0)

        for n in range(self.horizon_length):

            self.obs, self.expert = self.env_reset(self.done_indices)

            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs, self._rand_action_probs, beta_t, self.expert['actions'].to(self.ppo_device))
            self.experience_buffer.update_data('expert', n, self.expert['mus'].to(self.ppo_device))

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos, self.expert = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('rand_action_mask', n, res_dict['rand_action_mask'])

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            self.done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[self.done_indices])
            self.game_lengths.update(self.current_lengths[self.done_indices])
            self.algo_observer.process_infos(infos, self.done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            
            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos)
                
            self.done_indices = self.done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        return batch_dict


    def get_action_values(self, obs_dict, rand_action_probs, use_experts=0.0, expert=None):
        res_dict = super().get_action_values(obs_dict, rand_action_probs)
        num_envs = self.vec_env.env.task.num_envs
        expert_action_probs = to_torch([use_experts for _ in range(num_envs)], dtype=torch.float32, device=self.ppo_device)
        expert_action_probs = torch.bernoulli(expert_action_probs)
        det_action_mask = expert_action_probs == 1.0
        res_dict['actions'][det_action_mask] = expert[det_action_mask]
        res_dict['expert_mask'] = expert_action_probs

        return res_dict
    

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        expert = batch_dict['expert']
        expert_mask = batch_dict['expert_mask']
        self.dataset.values_dict['expert'] = expert
        self.dataset.values_dict['expert_mask'] = expert_mask
        return


    def _supervise_loss(self, student, teacher):
        e_loss = (student - teacher)**2

        info = {
            'expert_loss': e_loss.sum(dim=-1)
        }
        return info
    
    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos, expert = self.vec_env.step(actions)

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos, expert #.to(self.ppo_device)
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos, expert #.to(self.ppo_device)
        
    def env_reset(self, env_ids=None):
        obs, expert = self.vec_env.reset(env_ids)
        obs = self.obs_to_tensors(obs)
        return obs, expert


    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        expert_mus = input_dict['expert']
        obs_batch = self._preproc_obs(obs_batch)

        expert_mask = (input_dict['expert_mask'] > -1).float()
        expert_sum = torch.sum(expert_mask)

        rand_action_mask = input_dict['rand_action_mask']
        rand_action_sum = torch.sum(rand_action_mask)
        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            if rand_action_sum > 0:
                a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
                a_loss = a_info['actor_loss']
                a_clipped = a_info['actor_clipped'].float()
                c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
                c_loss = c_info['critic_loss']

                if self.epoch_num > 7000:
                    returns_var = return_batch.var(unbiased=False) + 1e-8  # avoid divide‑by‑0
                    errors_var = (return_batch - values).var(unbiased=False)
                    ev = 1.0 - errors_var / returns_var
                    self.ev_ma = 0.99 * self.ev_ma + 0.01 * ev.item()
                    if self.ev_ma >= 0.6:
                        self.critic_win_streak += 1
                    else:
                        self.critic_win_streak = 0
                        
                b_loss = self.bound_loss(mu)
                
                c_loss = torch.mean(c_loss)
                e_info = self._supervise_loss(mu, expert_mus)
                e_loss = e_info['expert_loss']
                a_loss = torch.sum(rand_action_mask * a_loss) / rand_action_sum
                entropy = torch.sum(rand_action_mask * entropy) / rand_action_sum
                b_loss = torch.sum(rand_action_mask * b_loss) / rand_action_sum
                a_clip_frac = torch.sum(rand_action_mask * a_clipped) / rand_action_sum
                e_loss = torch.mean(e_loss)
                if self.epoch_num > 6000 and self.critic_win_streak >= 3:
                    loss = a_loss * min((self.actor_update_num / 4000), 1) + self.critic_coef * c_loss + self.bounds_loss_coef * b_loss + self.expert_loss_coef * e_loss * max(1 - (self.actor_update_num / 4000), 0.1)
                    self.actor_update_num += 1
                elif self.epoch_num > 5000:
                    loss = min(((self.epoch_num - 5000) / 1000), 1) * self.critic_coef * c_loss + self.expert_loss_coef * e_loss
                else:
                    loss = self.expert_loss_coef * e_loss
                
            else:
                e_info = self._supervise_loss(mu, expert_mus)
                e_loss = e_info['expert_loss']
                e_loss = torch.mean(e_loss)
                loss = self.expert_loss_coef * e_loss
            
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
                    
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(e_info)
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        self.writer.add_scalar('losses/e_loss', torch_ext.mean_list(train_info['expert_loss']).item(), frame)

        return
