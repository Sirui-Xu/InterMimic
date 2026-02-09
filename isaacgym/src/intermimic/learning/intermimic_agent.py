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

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from . import a2c_common
import psutil
import subprocess
from isaacgym.torch_utils import *

import time
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn
import torch.distributed as dist
import os
from torch.nn.utils import clip_grad_norm_
from . import common_agent

from tensorboardX import SummaryWriter

class InterMimicAgent(common_agent.CommonAgent):
    def __init__(self, base_name, config):
        if config.get('multi_gpu', False):
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

            self.device_name = 'cuda:' + str(self.local_rank)
            config['device'] = self.device_name
            if self.rank != 0:
                config['print_stats'] = False
                config['lr_schedule'] = None
        super().__init__(base_name, config)

        if self._normalize_input:
            self._input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)
        self.resume_from = config['resume_from']
        self.done_indices = []
        self.epoch_num_start = 0
        return
    
    def trancate_gradients_and_step(self):
        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))

            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.world_size
                    )
                    offset += param.numel()

        if self.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # if self.has_central_value:
        #    self.central_value_net.update_lr(lr)
                
    def _maybe_init_ddp(self):
        """Init torch.distributed if we're in multi-GPU mode and it's not up yet."""
        if self.multi_gpu and dist.is_available() and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        # populate rank/world_size for convenience (works for single-GPU too)
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def _ddp_allreduce_scalar(self, val, op=dist.ReduceOp.SUM):
        """All-reduce a Python number (int/float) and return the reduced Python number."""
        if not (dist.is_available() and dist.is_initialized()):
            return val
        t = torch.tensor(float(val), device="cuda" if torch.cuda.is_available() else "cpu")
        dist.all_reduce(t, op=op)
        return t.item()

    def _sync_stats_ddp(self, train_info):
        """
        Drop-in replacement for self.hvd.sync_stats(self).
        - Aggregates numeric entries in train_info across ranks (mean by default).
        - You can extend this to sync any custom buffers (rewards, lengths) if needed.
        """
        if not (self.multi_gpu and dist.is_available() and dist.is_initialized()):
            return train_info

        # Average numeric stats in train_info across ranks
        world = float(self.world_size)
        for k, v in list(train_info.items()):
            if isinstance(v, (int, float)):
                summed = self._ddp_allreduce_scalar(v, op=dist.ReduceOp.SUM)
                train_info[k] = summed / world

        # (Optional) If you have buffers/trackers that need syncing, do it here.
        # Example stubs:
        # if hasattr(self.game_rewards, "sync_ddp"): self.game_rewards.sync_ddp()
        # if hasattr(self.game_lengths, "sync_ddp"): self.game_lengths.sync_ddp()

        return train_info

    def _ddp_average_value(self, x):
        """Average a scalar tensor across ranks (no-op if not distributed)."""
        if not (self.multi_gpu and dist.is_available() and dist.is_initialized()):
            return x
        t = x.detach().clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
        return t
    
    def train(self):
        if self.resume_from != 'None':
            try:
                self.restore(self.resume_from)
            except:
                print('Failed to restore from checkpoint')

        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.frame = 0

        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        model_output_file = os.path.join(self.nn_dir, self.config['name'])

        # --- NEW: init/process DDP world (no Horovod) ---
        self._maybe_init_ddp()

        self._init_train()
        if self.multi_gpu:
            torch.cuda.set_device(self.local_rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            if self.has_central_value:
                model_params.append(self.central_value_net.state_dict())
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])
            if self.has_central_value:
                self.central_value_net.load_state_dict(model_params[1])
                
        while True:
            epoch_num = self.update_epoch()
            train_info = self.train_epoch()  # core

            # --- NEW: sync stats across ranks (replacement for hvd.sync_stats) ---
            train_info = self._sync_stats_ddp(train_info)

            sum_time = train_info['total_time']
            total_time += sum_time
            frame = self.frame
            should_exit = False
            if self.epoch_num - self.epoch_num_start < 5:
                continue
            # Log/save only on rank 0
            if self.rank == 0:
                scaled_time = sum_time
                scaled_play_time = train_info['play_time']
                curr_frames = self.curr_frames

                # If each rank collects the same number of frames, the *global* frames this epoch is:
                #   global_curr_frames = curr_frames * self.world_size
                # If instead your code already accumulates global frames elsewhere, keep as-is.
                global_curr_frames = curr_frames * (self.world_size if self.multi_gpu else 1)
                self.frame += global_curr_frames

                if self.print_stats:
                    fps_step = global_curr_frames / scaled_play_time
                    fps_total = global_curr_frames / scaled_time
                    print(
                        f"epoch_num:{epoch_num} mean_rewards:{self._get_mean_rewards()} "
                        f"fps step: {fps_step:.1f} fps total: {fps_total:.1f}"
                    )

                self.writer.add_scalar('performance/total_fps', global_curr_frames / scaled_time, self.frame)
                self.writer.add_scalar('performance/step_fps',  global_curr_frames / scaled_play_time, self.frame)
                self.writer.add_scalar('info/epochs', epoch_num, self.frame)
                self._log_train_info(train_info, self.frame)

                self.algo_observer.after_print_stats(self.frame, epoch_num, total_time)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self._get_mean_rewards()
                    mean_lengths = self.game_lengths.get_mean()

                    for i in range(self.value_size):
                        self.writer.add_scalar(f'rewards{i}/frame', mean_rewards[i], self.frame)
                        self.writer.add_scalar(f'rewards{i}/iter',  mean_rewards[i], epoch_num)
                        self.writer.add_scalar(f'rewards{i}/time',  mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, self.frame)
                    self.writer.add_scalar('episode_lengths/iter',  mean_lengths, epoch_num)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                if self.save_freq > 0 and (epoch_num % self.save_freq == 0):
                    self.save(model_output_file)
                    if self._save_intermediate:
                        int_model_output_file = model_output_file + '_' + str(epoch_num).zfill(8)
                        self.save(int_model_output_file)

                if epoch_num > self.max_epochs:
                    self.save(model_output_file)
                    print('MAX EPOCHS NUM!')
                    should_exit = True

            # Optional: keep ranks in step before next epoch (helps around save/checkpoint)
            if self.multi_gpu and dist.is_available() and dist.is_initialized():
                dist.barrier()
                
            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.bool().item()
            
            if should_exit:
                return self.last_mean_rewards, epoch_num
        # not reached
        return

    def init_tensors(self):
        super().init_tensors()
        self._build_rand_action_probs()
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['rand_action_mask'] = torch.zeros(batch_shape, dtype=torch.float32, device=self.ppo_device)
        self.tensor_list += ['amp_obs', 'rand_action_mask']
        return
    
    def set_eval(self):
        super().set_eval()
        if self._normalize_input:
            self._input_mean_std.eval()
        return

    def set_train(self):
        super().set_train()
        if self._normalize_input:
            self._input_mean_std.train()
        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_input:
            state['amp_input_mean_std'] = self._input_mean_std.state_dict()
        
        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._normalize_input:
            self._input_mean_std.load_state_dict(weights['amp_input_mean_std'])
        return

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if self._normalize_input:
            self._input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        self.set_full_state_weights(checkpoint)

    def play_steps(self):
        self.set_eval()

        epinfos = []
        update_list = self.update_list

        for n in range(self.horizon_length):

            self.obs = self.env_reset(self.done_indices)

            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs, self._rand_action_probs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])

            invalid_obs = ~torch.isfinite(self.obs['obs'])  # True where obs is NaN or infinite
            invalid_batches = torch.any(invalid_obs, dim=1)  # Check if any invalid number in each batch (B, N)

            if torch.any(invalid_obs):
                print("invalid observation")
                print(torch.where(invalid_obs))
                self.obs['obs'][invalid_batches] = 0
            # Set self.dones to True for batches with invalid observations
                self.dones[invalid_batches] = True
                infos['terminate'][invalid_batches] = True

            shaped_rewards = self.rewards_shaper(rewards)
            # shaped_rewards = shaped_rewards * (((res_dict['actions'] - res_dict['mus'])**2).sum(dim=-1).mul(-0.01).exp().unsqueeze(-1))
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


    def get_action_values(self, obs_dict, rand_action_probs):
        processed_obs = self._preproc_obs(obs_dict['obs'])

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs_dict['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value

        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        
        rand_action_mask = torch.bernoulli(rand_action_probs)
        det_action_mask = rand_action_mask == 0.0
        res_dict['actions'][det_action_mask] = res_dict['mus'][det_action_mask]
        res_dict['rand_action_mask'] = rand_action_mask

        return res_dict


    # def get_action_values(self, obs):
    #     processed_obs = self._preproc_obs(obs['obs'])
    #     self.model.eval()
    #     input_dict = {
    #         'is_train': False,
    #         'prev_actions': None, 
    #         'obs' : processed_obs,
    #         'rnn_states' : self.rnn_states
    #     }

    #     with torch.no_grad():
    #         res_dict = self.model(input_dict)
    #         if self.has_central_value:
    #             states = obs['states']
    #             input_dict = {
    #                 'is_train': False,
    #                 'states' : states,
    #                 #'actions' : res_dict['action'],
    #                 #'rnn_states' : self.rnn_states
    #             }
    #             value = self.get_central_value(input_dict)
    #             res_dict['values'] = value
    #     if self.normalize_value:
    #         res_dict['values'] = self.value_mean_std(res_dict['values'], True)
    #     return res_dict
    

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        rand_action_mask = batch_dict['rand_action_mask']
        self.dataset.values_dict['rand_action_mask'] = rand_action_mask
        return
    
    def train_epoch(self):
        play_time_start = time.time()

        with torch.no_grad():
            batch_dict = self.play_steps_rnn() if self.is_rnn else self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None
        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / rnn_masks.nelement()
            print(frames_mask_ratio)

        epoch_kls = []

        for _ in range(self.mini_epochs_num):
            step_kls = []

            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if self.schedule_type == 'legacy':
                    kl_step = curr_train_info['kl']
                    if self.multi_gpu:
                        kl_step = self._ddp_average_value(kl_step)
                        curr_train_info['kl'] = kl_step
                    self.last_lr, self.entropy_coef = self.scheduler.update(
                        self.last_lr, self.entropy_coef, self.epoch_num, 0, kl_step.item()
                    )
                    self.update_lr(self.last_lr)

                if train_info is None:
                    train_info = {k: [v] for k, v in curr_train_info.items()}
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

                step_kls.append(curr_train_info['kl'])

            av_kls = torch_ext.mean_list(step_kls)   # <---- back to torch_ext
            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self._ddp_average_value(av_kls)
                self.last_lr, self.entropy_coef = self.scheduler.update(
                    self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
                )
                self.update_lr(self.last_lr)

            epoch_kls.append(av_kls)

        if self.schedule_type == 'standard_epoch':
            epoch_av_kl = torch_ext.mean_list(epoch_kls)  # <---- again use torch_ext
            if self.multi_gpu:
                epoch_av_kl = self._ddp_average_value(epoch_av_kl)
            self.last_lr, self.entropy_coef = self.scheduler.update(
                self.last_lr, self.entropy_coef, self.epoch_num, 0, epoch_av_kl.item()
            )
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        self._record_train_batch_info(batch_dict, train_info)

        return train_info

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
        obs_batch = self._preproc_obs(obs_batch)
        
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

        # --- zero grads (DDP/AMP-safe) ---
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']
            a_clipped = a_info['actor_clipped'].float()

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']

            b_loss = self.bound_loss(mu)
            
            c_loss = self._loss_mean(c_loss)
            a_loss = self._loss_mean(a_loss) 
            entropy = self._loss_mean(entropy)
            b_loss = self._loss_mean(b_loss)
            a_clip_frac = self._loss_mean(a_clipped)
            
            loss = a_loss + self.critic_coef * c_loss + self.bounds_loss_coef * b_loss # - entropy * self.entropy_coef
            
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
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)

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

        return

    def _ddp_allreduce_sum(self, t):
        """All-reduce sum for a 0-D or 1-D tensor; returns a tensor."""
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t

    def _loss_mean(self, c_unreduced):
        c_sum   = c_unreduced.reshape(-1).float().sum()
        c_count = torch.tensor([c_unreduced.numel()], device=c_sum.device, dtype=torch.float32).sum()
        if dist.is_available() and dist.is_initialized():
            c_sum   = self._ddp_allreduce_sum(c_sum)
            c_count = self._ddp_allreduce_sum(c_count)
        c_loss = c_sum / c_count.clamp_min(1.0)
        return c_loss
    
    def _load_config_params(self, config):
        super()._load_config_params(config)
        
        # when eps greedy is enabled, rollouts will be generated using a mixture of
        # a deterministic and stochastic actions. The deterministic actions help to
        # produce smoother, less noisy, motions that can be used to train a better
        # discriminator. If the discriminator is only trained with jittery motions
        # from noisy actions, it can learn to phone in on the jitteriness to
        # differential between real and fake samples.
        self._enable_eps_greedy = bool(config['enable_eps_greedy'])
        self._amp_observation_space = self.env_info['amp_observation_space']
        self._normalize_input = config.get('normalize_input', True)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        config['amp_input_shape'] = self._amp_observation_space.shape
        return config
    
    def _build_rand_action_probs(self):
        num_envs = self.vec_env.env.task.num_envs
        env_ids = to_torch(np.arange(num_envs), dtype=torch.float32, device=self.ppo_device)

        self._rand_action_probs = 1.0 - torch.exp(10 * (env_ids / (num_envs - 1.0) - 1.0))
        self._rand_action_probs[0] = 1.0
        self._rand_action_probs[-1] = 0.0
        
        if not self._enable_eps_greedy:
            self._rand_action_probs[:] = 1.0

        return

    def _init_train(self):
        super()._init_train()
        return

    
    def _calc_advs(self, batch_dict):
        returns = batch_dict['returns']
        values = batch_dict['values']
        rand_action_mask = batch_dict['rand_action_mask']

        advantages = returns - values
        advantages = torch.sum(advantages, axis=1)
        if self.normalize_advantage:
            advantages = torch_ext.normalization_with_masks(advantages, rand_action_mask)

        return advantages
    
    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        return
    
    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=1)

    def get_cpu_memory_usage(self):
        return psutil.virtual_memory().percent
    
    # Function to get GPU usage
    def get_gpu_usage(self):
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader'], stdout=subprocess.PIPE)
        return int(result.stdout.decode().strip().split()[0])
    
    # Function to get GPU memory usage
    def get_gpu_memory_usage(self):
        return torch.cuda.memory_allocated() / 1e9

    def _log_train_info(self, train_info, frame):
        self.writer.add_scalar('performance/update_time', train_info['update_time'], frame)
        self.writer.add_scalar('performance/play_time', train_info['play_time'], frame)
        self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(train_info['actor_loss']).item(), frame)
        self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(train_info['critic_loss']).item(), frame)
        
        self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(train_info['b_loss']).item(), frame)
        self.writer.add_scalar('losses/entropy', torch_ext.mean_list(train_info['entropy']).item(), frame)
        self.writer.add_scalar('info/last_lr', train_info['last_lr'][-1] * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/lr_mul', train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * train_info['lr_mul'][-1], frame)
        self.writer.add_scalar('info/clip_frac', torch_ext.mean_list(train_info['actor_clip_frac']).item(), frame)
        self.writer.add_scalar('info/kl', torch_ext.mean_list(train_info['kl']).item(), frame)

        self.writer.add_scalar('usage/cpu', self.get_cpu_usage(), frame)
        self.writer.add_scalar('usage/gpu', self.get_gpu_usage(), frame)
        self.writer.add_scalar('usage/cpu_memory', self.get_cpu_memory_usage(), frame)
        self.writer.add_scalar('usage/gpu_memory', self.get_gpu_memory_usage(), frame)

        return