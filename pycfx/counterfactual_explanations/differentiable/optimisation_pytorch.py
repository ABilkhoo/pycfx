"""
pycfx/counterfactual_explanations/differentiable/optimisation_pytorch.py
PyTorch implemenation of a DifferentiableOptimisation loop.
"""

from pycfx.models.latent_encodings import IdentityEncoding
from pycfx.counterfactual_explanations.differentiable.optimisation_loop import DifferentiableOptimisation, OptimisationState, register_optimisation_loop
from pycfx.helpers.constants import BACKEND_PYTORCH

import torch

class SalientFeatureOptimizer(torch.optim.Optimizer):
    """
    PyTorch implementation of the JSMA-like optimiser used by Schut et al. "Generating interpretable counterfactual explanations by implicit minimisation of epistemic and aleatoric uncertainties." (2021).
    """
    
    def __init__(self, params, lr=0.1):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                grad_flat = grad.view(-1)
                if grad_flat.numel() == 0:
                    continue

                idx = torch.argmax(grad_flat.abs())
                update_val = -1 * lr * torch.sign(grad_flat[idx])

                grad_update = torch.zeros_like(grad)
                grad_update.view(-1)[idx] = update_val

                p.add_(grad_update)

        return loss

@register_optimisation_loop(BACKEND_PYTORCH)
class PyTorchMLP_Optimisation(DifferentiableOptimisation):

    def setup(self, x_factual, y_target):
        """
        Helper to return the torch `device`, `optimiser`, and initial `opt_state` based on `x_factual` and `y_target`
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        torch.manual_seed(0)

        x_factual = torch.from_numpy(x_factual).float().to(device)
        y_factual = self.model.pytorch_model(x_factual)

        if self.input_properties.y_onehot:
            y_target = torch.nn.functional.one_hot(torch.tensor(y_target), self.input_properties.n_targets).float().to(device)
        else:
            y_target = torch.from_numpy(y_target).float().to(device)

        z = torch.autograd.Variable(self.latent_encoding.encode(x_factual.clone()), requires_grad=True)
        z_factual = z.clone()
        x = self.latent_encoding.decode(z)
        x_enc = self.fix_encoding(x)

        if self.jsma:
            optimiser = SalientFeatureOptimizer([z], self.lr)
        else:
            optimiser = torch.optim.Adam([z], self.lr, amsgrad=True)
        
        y_enc = self.model.pytorch_model(x_enc).to(device)

        opt_state = OptimisationState(self.model, z, z_factual, x_enc, y_enc, x_factual, y_factual, y_target, 0, self.n_iter)

        return device, optimiser, opt_state
    
    def is_correct_classification(self, y_enc, y_target):
        """
        Helper to check whether the prediction of the datapoint being optimised is a valid CFX, i.e. it predicts the target class
        """

        return torch.argmax(y_enc) == torch.argmax(y_target)
    
    def fix_encoding(self, x):
        """
        Helper to fix the encoding of the optimised datapoint to be within bounds, or a valid categorically or ordinally encoded variable. 
        If a latent space is used for the optimisation, no change is made to x.
        """

        if not isinstance(self.latent_encoding, IdentityEncoding) :
            return x

        x_enc = torch.zeros_like(x)

        for i in range(self.input_properties.n_features):
            feature_class = self.input_properties.feature_classes[i]
            bound = self.input_properties.bound[i]

            if feature_class == 'numeric' and bound is not None:
                proj = torch.clamp(x[i], bound[0], bound[1])

            elif feature_class == 'ordinal' or feature_class == 'ordinal_normalised':
                if self.tensor_bounds is None:
                    self.tensor_bounds = []
                    for j in range(self.input_properties.n_features):
                        if self.input_properties.bound[j]:
                            self.tensor_bounds.append(torch.tensor(self.input_properties.bound[j], device=x.device))
                        else:
                            self.tensor_bounds.append(None)

                diffs = (x[i] - self.tensor_bounds[i]) ** 2
                idx = torch.argmin(diffs, dim=-1)
                proj = bound[idx]

            x_enc[i] = proj

        for group in self.input_properties.categorical_groups:
            group_vals = x[group]

            idx = torch.argmax(group_vals, dim=-1)
            onehot = torch.functional.F.one_hot(idx, num_classes=group_vals.shape[-1])
            proj = onehot.to(group_vals.dtype)

            x_enc[group] = proj

        return x_enc


    def optimise_minmax(self, x, y_target, n_it_outer=2):
        device, optimiser, opt_state = self.setup(x, y_target)
        y_target = opt_state.y_target

        min_max_lambda = torch.tensor(self.min_max_lambda).float().to(device)

        prev_solution = None
        change = torch.inf

        while not (self.early_stopping and self.is_correct_classification(opt_state.y_enc, y_target)) and n_it_outer > 0:
            opt_state.it = 0
            while not (self.early_stopping and self.is_correct_classification(opt_state.y_enc, y_target)) and opt_state.it < self.n_iter:
                optimiser.zero_grad()
                
                loss_term_0 = self.losses[0].loss(opt_state) * self.losses_weights[0]

                remaining_losses = torch.tensor(0.0, requires_grad=True)

                for i in range(1, len(self.losses)):
                    remaining_losses = remaining_losses + self.losses[i].loss(opt_state) * self.losses_weights[i]

                loss = loss_term_0 + min_max_lambda * remaining_losses
                loss.backward(retain_graph=self.retain_graph)
                optimiser.step()
                
                x = self.latent_encoding.decode(opt_state.z)
                opt_state.x_enc = self.fix_encoding(x)
                opt_state.y_enc = self.model.pytorch_model(opt_state.x_enc).to(device)


                if prev_solution is not None:
                    change = torch.norm(opt_state.x_enc - prev_solution).item()
                if change < 1e-4:  # Threshold for minimal change
                    # print("Early stopping")
                    break

                prev_solution = opt_state.x_enc.clone()

                opt_state.it += 1
                
            n_it_outer -= 1
            min_max_lambda -= 0.05

        return opt_state.x_enc.cpu().detach().numpy()

    def optimise_min(self, x, y_target):
        device, optimiser, opt_state = self.setup(x, y_target)
        y_target = opt_state.y_target
        prev_solution = None
        change = torch.inf

        while (not (self.early_stopping and self.is_correct_classification(opt_state.y_enc, y_target))) and opt_state.it < self.n_iter:
            optimiser.zero_grad()
            
            losses = 0.0

            for i in range(0, len(self.losses)):
                losses = losses + self.losses[i].loss(opt_state) * self.losses_weights[i]
            
            # losses *= -1

            losses.backward(retain_graph=self.retain_graph)

            optimiser.step()
            
            x = self.latent_encoding.decode(opt_state.z)
            opt_state.x_enc = self.fix_encoding(x)
            opt_state.y_enc = self.model.pytorch_model(opt_state.x_enc).to(device)


            if prev_solution is not None:
                change = torch.norm(opt_state.x_enc - prev_solution).item()
            if change < 1e-4:  # Threshold for minimal change
                # print("Early stopping")
                break

            prev_solution = opt_state.x_enc.clone()
            opt_state.it += 1

        return opt_state.x_enc.cpu().detach().numpy()


    