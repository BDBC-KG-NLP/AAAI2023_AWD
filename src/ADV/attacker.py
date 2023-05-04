import torch
import torch.nn as nn
import sys

class Attacker:

    def __init__(self, args):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

    @torch.enable_grad()
    def __call__(self, inputs, targets, model, epsilon):
        inputs['inputs_embeds'].requires_grad_(True)

        output = model(**inputs)
        logits = output[1]
        ce_loss = self.criterion(logits, targets)
        inputs_embeds = inputs['inputs_embeds']
        grad_adv = torch.autograd.grad(outputs=ce_loss,
                                       inputs=inputs_embeds,
                                       grad_outputs=ce_loss.new_ones(ce_loss.size()),
                                       retain_graph=True, # for loss computation
                                       create_graph=False,
                                       only_inputs=True)[0]
        if self.args.normalize:
            grad_adv = grad_adv.view(inputs_embeds.size(0), -1)
            grad_adv = grad_adv / (grad_adv.norm(dim=1, keepdim=True) + 1e-12)
            grad_adv = grad_adv.view_as(inputs_embeds)
        delta = grad_adv
        if self.args.norm_type == 'linf':
            delta = delta.sign().clamp(-epsilon, epsilon)
        elif self.args.norm_type == 'l2':
            delta = delta.view(inputs_embeds.size(0), -1)
            delta = epsilon * delta / (delta.norm(dim=1, keepdim=True) + 1e-12)
            delta = delta.view_as(inputs_embeds)
        adv_inputs = inputs_embeds + delta
        return adv_inputs.detach(), ce_loss