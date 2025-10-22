"""
pycfx/counterfactual_explanations/differentiable/optimisation_pytorch.py
Keras implemenation of a DifferentiableOptimisation loop.
"""

from pycfx.models.latent_encodings import IdentityEncoding
from pycfx.counterfactual_explanations.differentiable.optimisation_loop import DifferentiableOptimisation, OptimisationState, register_optimisation_loop
from pycfx.helpers.constants import BACKEND_TENSORFLOW

import tensorflow as tf
import numpy as np
import keras

class SalientFeatureOptimizer_TensorFlow(tf.keras.optimizers.Optimizer):
    """
    TensorFlow implementation of the JSMA-like optimiser used by Schut et al. "Generating interpretable counterfactual explanations by implicit minimisation of epistemic and aleatoric uncertainties." (2021).
    """

    def __init__(self, learning_rate=0.1, name="SalientFeatureOptimizer", **kwargs):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)


    def update_step(self, grad, var, lr):

        grad_flat = tf.reshape(grad, [-1])
        idx = tf.argmax(tf.abs(grad_flat))

        grad_update = tf.scatter_nd(
            indices=[[idx]],
            updates=[-1 * lr * tf.sign(grad_flat[idx])],
            shape=tf.shape(grad_flat),
        )

        grad_update = tf.reshape(grad_update, tf.shape(var))
        var.assign_add(grad_update)

    def _resource_apply_sparse(self, grad, var, indices):
        dense_grad = tf.convert_to_tensor(tf.IndexedSlices(grad, indices))
        return self._resource_apply_dense(dense_grad, var)


@register_optimisation_loop(BACKEND_TENSORFLOW)
class KerasMLP_Optimisation(DifferentiableOptimisation):

    def get_model_prediction(self, x):
        """
        Helper to obtain the prediction for a single example x.
        """

        x = tf.expand_dims(x, 0)
        y = self.model.keras_model(x)
        return tf.squeeze(y, 0)

    def setup(self, x_factual, y_target):
        """
        Helper to return the torch `device`, `optimiser`, and initial `opt_state` based on `x_factual` and `y_target`
        """

        x_factual = tf.convert_to_tensor(x_factual, dtype=tf.float32)
        y_factual = self.get_model_prediction(x_factual)

        if self.input_properties.y_onehot:
            y_target = tf.one_hot(y_target, depth=self.input_properties.n_features)
        else:
            y_target = tf.convert_to_tensor(y_target)

        z = tf.Variable(self.latent_encoding.encode(x_factual))
        z_factual = tf.Variable(z.value())
        x = self.latent_encoding.decode(z)
        x_enc = self.fix_encoding(x)

        if self.jsma:
            optimiser = SalientFeatureOptimizer_TensorFlow(self.lr)
        else:
            optimiser = keras.optimizers.Adam(learning_rate=self.lr)
        
        y_enc = self.get_model_prediction(x_enc)

        opt_state = OptimisationState(self.model, z, z_factual, x_enc, y_enc, x_factual, y_factual, y_target, 0, self.n_iter)

        return optimiser, opt_state
    
    def is_correct_classification(self, y_enc, y_target):
        """
        Helper to check whether the prediction of the datapoint being optimised is a valid CFX, i.e. it predicts the target class
        """

        return tf.argmax(y_enc) == tf.argmax(y_target)
    
    def fix_encoding(self, x):
        """
        Helper to fix the encoding of the optimised datapoint to be within bounds, or a valid categorically or ordinally encoded variable. 
        If a latent space is used for the optimisation, no change is made to x.
        """

        if not isinstance(self.latent_encoding, IdentityEncoding) :
            return x

        x_enc = tf.zeros_like(x)
        update = []
        update_vals = []

        for i in range(self.input_properties.n_features):
            feature_class = self.input_properties.feature_classes[i]
            bound = self.input_properties.bound[i]

            if feature_class == 'numeric' and bound is not None:
                lb = bound[0]
                ub = bound[1]
                if np.isinf(lb):
                    lb = tf.float32.min
                if np.isinf(ub):
                    ub = tf.float32.max

                update.append([i])
                update_vals.append(tf.clip_by_value(x[i], lb, ub))

            elif feature_class == 'ordinal' or feature_class == 'ordinal_normalised':
                if self.tensor_bounds is None:
                    self.tensor_bounds = []
                    for j in range(self.input_properties.n_features):
                        if self.input_properties.bound[j]:
                            self.tensor_bounds.append(tf.constant(self.input_properties.bound[j]))
                        else:
                            self.tensor_bounds.append(None)

                diffs = (x[i] - self.tensor_bounds[i]) ** 2
                idx = tf.argmin(diffs, axis=-1)
                
                update.append([i])
                update_vals.append(bound[idx])

        x_enc = tf.tensor_scatter_nd_update(x_enc, update, update_vals)

        for group in self.input_properties.categorical_groups:
            group_vals = tf.gather(x, group)
            idx = tf.argmax(group_vals, axis=-1)
            onehot = tf.one_hot(idx, depth=group_vals.shape[-1], dtype=group_vals.dtype)

            idxs = np.array(group).reshape(-1, 1)
            tf.tensor_scatter_nd_update(x_enc, idxs, onehot)

        return x_enc
  
    def optimise_min(self, x, y_target):
        optimiser, opt_state = self.setup(x, y_target)
        y_target = opt_state.y_target
        prev_solution = None
        change = np.inf
        
        while (not (self.early_stopping and self.is_correct_classification(opt_state.y_enc, y_target))) and opt_state.it < self.n_iter:
            with tf.GradientTape() as tape:
                tape.watch(opt_state.z)
                
                x = self.latent_encoding.decode(opt_state.z)
                opt_state.x_enc = self.fix_encoding(x)
                opt_state.y_enc = self.get_model_prediction(opt_state.x_enc)

                losses = tf.constant(0.0)
                tape.watch(losses)

                for i in range(0, len(self.losses)):
                    losses += self.losses[i].loss(opt_state) * self.losses_weights[i]
            
            grads = tape.gradient(losses, opt_state.z)
            optimiser.apply_gradients([(grads, opt_state.z)])

            if prev_solution is not None:
                change = tf.norm(opt_state.x_enc - prev_solution)
            if change < 1e-4:  
                # print("Early stopping")
                break

            prev_solution = tf.constant(opt_state.x_enc)
            opt_state.it += 1

        return opt_state.x_enc.numpy()
    
  
    def optimise_minmax(self, x, y_target, n_it_outer=2):
        optimiser, opt_state = self.setup(x, y_target)
        y_target = opt_state.y_target
        prev_solution = None
        change = np.inf

        while not (self.early_stopping and self.is_correct_classification(opt_state.y_enc, y_target)) and n_it_outer > 0:
            opt_state.it = 0
            while not (self.early_stopping and self.is_correct_classification(opt_state.y_enc, y_target)) and opt_state.it < self.n_iter:
                # loss_value = self.calculate_losses_minmax(opt_state, optimiser)
                with tf.GradientTape() as tape:
                    tape.watch(opt_state.z)

                    x = self.latent_encoding.decode(opt_state.z)
                    opt_state.x_enc = self.fix_encoding(x)
                    opt_state.y_enc = self.get_model_prediction(opt_state.x_enc)

                    loss_term_0 = self.losses[0].loss(opt_state) * self.losses_weights[0]
                    remaining_losses = tf.constant(0.0)

                    for i in range(1, len(self.losses)):
                        remaining_losses = remaining_losses + self.losses[i].loss(opt_state) * self.losses_weights[i]

                    loss = loss_term_0 + self.min_max_lambda * remaining_losses 
                
                grads = tape.gradient(loss, opt_state.z)
                optimiser.apply_gradients([(grads, opt_state.z)])

                if prev_solution is not None:
                    change = tf.norm(opt_state.x_enc - prev_solution)
                if change < 1e-4:  # Threshold for minimal change
                    # print("Early stopping")
                    break

                prev_solution = tf.constant(opt_state.x_enc)

                opt_state.it += 1
                
            n_it_outer -= 1
            self.min_max_lambda -= 0.05

        return opt_state.x_enc.numpy()