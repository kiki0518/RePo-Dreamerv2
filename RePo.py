import torch
import torch.nn as nn
from torch.distributions import kl_divergence

import tools
import models
import exploration as expl
from dreamer import Dreamer

class RePoWorldModel(models.WorldModel):
    """WorldModel with RePo-style KL constraint via dual variable beta."""

    def __init__(self, step, config):
        super().__init__(step, config)
        # Dual variable (log-space)
        self._log_beta = nn.Parameter(
            torch.tensor(float(config.init_beta)).log()
        )
        # Beta optimizer
        self._beta_opt = tools.Optimizer(
            'beta', [self._log_beta],
            config.beta_lr, config.opt_eps,
            config.grad_clip, config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        # RePo specific settings
        self._target_kl = config.target_kl
        # balance the prior and posterior KL weights (corresponds to α in the paper)
        self._kl_balance = config.prior_train_steps / (1 + config.prior_train_steps)

    def _train(self, data):
        data = self.preprocess(data)
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(embed, data['action'])

                # seperatly compute KL for training prior and posterior
                dist_post = self.dynamics.get_dist(post)
                dist_prior = self.dynamics.get_dist(prior)
                kl_prior = kl_divergence(dist_post.detach(), dist_prior)     # update prior
                kl_post = kl_divergence(dist_post, dist_prior.detach())     # update posterior
                kl_mix = self._kl_balance * kl_prior + (1 - self._kl_balance) * kl_post
                kl_value = kl_mix.mean()  # scalar for constraint

                # RePo KL constraint: L_kl = beta * (KL - target)
                kl_violation = kl_value - self._target_kl
                kl_loss = self._log_beta.exp().detach() * kl_violation

                losses = {}
                likes = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    like = pred.log_prob(data[name])
                    likes[name] = like
                    losses[name] = -like.mean() * self._scales.get(name, 1.0)

                model_loss = sum(losses.values()) + kl_loss

            # update world model parameters first
            model_metrics = self._model_opt(model_loss, self.parameters())

            # then update the dual variable beta
            beta_loss = -self._log_beta * kl_violation.detach()
            beta_metrics = self._beta_opt(beta_loss)

        metrics = {}
        metrics.update({f'{k}_loss': tools.to_np(v) for k, v in losses.items()})
        metrics.update(model_metrics)
        metrics.update(beta_metrics)
        metrics['kl'] = tools.to_np(kl_value)
        metrics['kl_violation'] = tools.to_np(kl_violation)
        metrics['beta'] = tools.to_np(self._log_beta.exp())
        metrics['beta_loss'] = tools.to_np(beta_loss)

        with torch.cuda.amp.autocast(self._use_amp):
            metrics['prior_ent'] = tools.to_np(self.dynamics.get_dist(prior).entropy().mean())
            metrics['post_ent'] = tools.to_np(self.dynamics.get_dist(post).entropy().mean())

        context = dict(
            embed=embed,
            feat=self.dynamics.get_feat(post),
            kl=kl_mix,
            postent=self.dynamics.get_dist(post).entropy(),
        )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics


class RePo(Dreamer):
    """Dreamer agent variant that swaps WorldModel to RePoWorldModel."""

    def __init__(self, config, logger, dataset):
        super().__init__(config, logger, dataset)
        # replace WorldModel with RePoWorldModel
        self._wm = RePoWorldModel(self._step, config).to(config.device)

        # re-initialize task behavior(since the previous one used the old wm)
        self._task_behavior = models.ImagBehavior(
            config, self._wm, config.behavior_stop_grad
        )
        reward = lambda f, s, a: self._wm.heads['reward'](f).mean
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]()
