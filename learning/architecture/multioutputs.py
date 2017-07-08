import copy
import types
import pickle

import numpy

from learning import Model
from learning.rlearn import RLTable

class MultiOutputs(Model):
    """Ensemble enabling given model to return a higher dimensional output tensor.

    Can be used to return an output vector from a model returing an output value.
    Or an output matrix from a model returning an output vector.
    Etc.
    MultiOutputs can be nested.

    For each output value,
    a model mapping input_vec to output value is learned.
    New outputs are returned by concatenating all model outputs.

    Args:
        models: list<Model> or Model; List of models,
            or Model that is duplicated by num_outputs
        num_outputs: How many components in target vectors.
    """
    def __init__(self, models, num_outputs=None):
        super(MultiOutputs, self).__init__()

        if isinstance(models, Model):
            # Store copy of model for each output
            if num_outputs is None:
                raise ValueError('If Model is given, num_outputs must not be None')
            self._models = [copy.deepcopy(models) for _ in range(num_outputs)]
        else:
            if not isinstance(models, (list, tuple)):
                raise ValueError('models must be list or tuple, or Model')

            # Validate that list contains Model's, and no duplicates
            for i, model in enumerate(models):
                if not isinstance(model, Model):
                    raise ValueError('models must contain instances of Model')

                # No duplicates
                for other_model in models[i+1:]:
                    if other_model is model:
                        raise ValueError('models should not contain duplicate instances')

            # Validation done, store it
            self._models = models[:]

        self._num_outputs = len(self._models)

        # Use reinforcement learning to select which output to update
        # We use different between new and old error as reward
        self._rl_agent = RLTable([None], range(self._num_outputs),
                                 initial_reward=1.0, update_rate=0.25, reward_growth=0.01)
        self._errors = [None]*self._num_outputs

    def reset(self):
        """Reset this model."""
        # Reset RL agent
        self._rl_agent = RLTable([None], range(self._num_outputs))
        self._errors = [None]*self._num_outputs

        # Reset each stored model
        for model in self._models:
            model.reset()

    def activate(self, inputs):
        """Return the model outputs for given inputs.

        One output for each stored model.
        """
        return [model.activate(inputs) for model in self._models]

    def train_step(self, input_matrix, target_matrix):
        """Adjust the model towards the targets for given inputs.

        Adjusts each AUTO unit towards its corresponding target vector.
        """
        if len(target_matrix[0]) != self._num_outputs:
            raise ValueError('Target matrix column does not match expected number of outputs')

        # Each model learns a single component (column) of the target matrix
        # First iteration, we update all outputs (for baseline)
        # Then we update only a single output value per iteration.
        # NOTE: If a stored model doesn't return error, we default to updating all
        # every iteration, because we can't know which is best to update
        #   TODO: Update None errors every iteration, and select one from non-Nones
        if None in self._errors:
            self._update_all_outputs(input_matrix, target_matrix)
        else:
            self._update_one_output(input_matrix, target_matrix)

        try:
            return sum(self._errors) / len(self._errors)
        except TypeError:
            # Model didn't return error
            return None

    def train(self, input_matrix, target_matrix, *args, **kwargs):
        """Train model to converge on set of patterns."""
        if len(target_matrix[0]) != self._num_outputs:
            raise ValueError('Target matrix column does not match expected number of outputs')

        # Train each stored model
        for i, (model, targets) in enumerate(zip(self._models, _transpose_rowcol(target_matrix))):
            if self.logging:
                if i != 0:
                    print
                print 'Training Model %d:' % (i+1)
            else:
                model.logging = self.logging
            model.train(input_matrix, targets, *args, **kwargs)

        self.iteration = sum([model.iteration for model in self._models])

    def serialize(self):
        """Convert model into string.

        Returns:
            string; A string representing this network.
        """
        # Use serialize on each model, instead of pickle
        serialized_models = [(type(model), model.serialize()) for model in self._models]

        # Pickle all other attributes
        attributes = copy.copy(self.__dict__)
        del attributes['_models']

        return pickle.dumps((serialized_models, attributes), protocol=2)

    @classmethod
    def unserialize(cls, serialized_model):
        """Convert serialized model into Model.

        Returns:
            Model; A Model object.
        """
        serialized_models, attributes = pickle.loads(serialized_model)

        # Make model, from serialized models and attributes
        model = MultiOutputs.__new__(MultiOutputs)
        model.__dict__ = attributes

        # unserialize each model
        model._models = [class_.unserialize(model_str) for class_, model_str in serialized_models]

        return model

    def mse(self, input_vec, target_vec):
        """Return the mean squared error (MSE) for a pattern."""
        return numpy.mean([model.mse(input_vec, target)
                           for model, target in zip(self._models, _transpose_rowcol(target_vec))])

    def _update_one_output(self, input_matrix, target_matrix):
        """Update the model that most shows the ability to improve."""
        # Use reinforcement learning to select output to update.
        to_update = self._rl_agent.get_action(None)
        new_error = self._models[to_update].train_step(
            input_matrix, _matrix_col(target_matrix, to_update))

        # Update RL agent
        self._rl_agent.update(None, to_update,
                              _get_reward(self._errors[to_update], new_error))

        # Update our error list
        self._errors[to_update] = new_error

    def _update_all_outputs(self, input_matrix, target_matrix):
        """Update all stored models."""
        for i, (model, targets) in enumerate(zip(self._models, _transpose_rowcol(target_matrix))):
            self._errors[i] = model.train_step(input_matrix, targets)

def _get_reward(old_error, new_error):
    """Return RL agent reward.

    Reward for RL agent is difference between new and previous error for output.
    Plus small amount for error (prioritize higher error)
    """
    return (old_error-new_error) + 0.2*new_error

def _matrix_col(matrix, i):
    """Return the ith column of matrix."""
    if isinstance(matrix, numpy.ndarray):
        return _np_matrix_col(matrix, i)

    # List of list
    if isinstance(matrix[0], (list, tuple, numpy.ndarray)):
        return [row[i] for row in matrix]
    else:
        # Only 1d, take row
        return matrix[i]

def _np_matrix_col(matrix, i):
    """Return the ith column of matrix."""
    if len(matrix.shape) == 1:
        # Only 1d, take row
        return matrix[i]

    return matrix[:, i]

def _transpose_rowcol(matrix):
    """Return matrix with row and col swapped.

    Other axis are left intact.
    """
    if isinstance(matrix, numpy.ndarray):
        return _np_transpose_rowcol(matrix)

    # List of list
    if isinstance(matrix[0], (list, tuple, numpy.ndarray)):
        return zip(*matrix)
    else:
        # Only 1d, no change
        return matrix

def _np_transpose_rowcol(matrix):
    """Return matrix with row and col swapped.

    Other axis are left intact.
    """
    if len(matrix.shape) == 1:
        # Only 1d, no change
        return matrix

    return matrix.transpose([1, 0] + range(len(matrix.shape))[2:])
