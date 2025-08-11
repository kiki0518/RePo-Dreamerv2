import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Categorical

import tools
import models
import exploration as expl
from torch import distributions as torchd
from dreamer import Dreamer

class RePoWorldModel(models.WorldModel):
    """WorldModel with RePo-style KL constraint via dual variable beta."""

    def __init__(self, step, config):
        super().__init__(step, config)
        self._log_beta = nn.Parameter(
            torch.tensor(float(config.init_beta)).log()
        )
        self._beta_opt = tools.Optimizer(
            'beta', [self._log_beta],
            config.beta_lr, config.opt_eps,
            config.grad_clip, config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
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
                dist_post = torchd.independent.Independent(tools.OneHotDist(logits = post["logit"]), 1)
                dist_prior = torchd.independent.Independent(tools.OneHotDist(logits = prior["logit"]), 1)

                dist_post_detached = torchd.independent.Independent(tools.OneHotDist(logits=dist_post.base_dist.logits.detach()), 1)
                dist_prior_detached = torchd.independent.Independent(tools.OneHotDist(logits=dist_prior.base_dist.logits.detach()), 1)

                kl_prior = kl_divergence(dist_post_detached, dist_prior).mean()
                kl_post = kl_divergence(dist_post, dist_prior_detached).mean()

                kl_alpha = self._config.prior_train_steps / (1 + self._config.prior_train_steps)
                kl_value = kl_alpha * kl_prior + (1 - kl_alpha) * kl_post
                kl_violation = kl_value - self._target_kl
                kl_loss = self._log_beta.exp().detach() * kl_violation

                # assert not torch.isnan(kl_value), f"[NaN] kl_value: {kl_value}"
                # assert not torch.isnan(self._log_beta), f"[NaN] log_beta: {self._log_beta}"
                # assert not torch.isnan(kl_loss), f"[NaN] kl_loss: {kl_loss}"

                losses = {}
                likes = {}
                for name, head in self.heads.items():
                    grad_head = (name in self._config.grad_heads) and name != 'image'
                    # observation loss needs to be detached
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    like = pred.log_prob(data[name])
                    likes[name] = like
                    loss = -like.mean() * self._scales.get(name, 1.0)

                    # assert not torch.isnan(loss), f"[NaN] loss for {name}: {loss}"
                    losses[name] = loss

                model_loss = sum(losses.values()) + kl_loss
                # assert not torch.isnan(model_loss), f"[NaN] model_loss: {model_loss}"

            # update world model parameters first
            model_metrics = self._model_opt(model_loss, self.parameters())
            # then update the dual variable beta
            beta_loss = -self._log_beta * kl_violation.detach()
            # beta_metrics = self._beta_opt(beta_loss)
            # assert not torch.isnan(beta_loss), f"[NaN] beta_loss: {beta_loss}"

            self._beta_opt.zero_grad()
            beta_loss.backward()
            self._beta_opt.step()

            # Apply clipping to ensure beta <= 10
            with torch.no_grad():
                beta = self._log_beta.exp()
                beta = torch.clamp(beta, max=10.0)
                self._log_beta.data = torch.log(beta)

        metrics = {}
        metrics.update({f'{k}_loss': tools.to_np(v) for k, v in losses.items()})
        metrics.update(model_metrics)
        # metrics.update(beta_metrics)
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
                kl=kl_value,
                postent=tools.to_np(self.dynamics.get_dist(post).entropy()),
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
