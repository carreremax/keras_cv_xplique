"""
Custom tensorflow operator for Attributions
"""

#import inspect
import tensorflow as tf

from xplique.types import Callable, Optional
#from xplique.commons.exceptions import raise_invalid_operator, no_gradients_available
from xplique.commons.callable_operations import predictions_one_hot_callable
from xplique.commons.operators import check_operator, predictions_operator, operator_batching, \
    batch_predictions

batch_predictions_one_hot_callable = operator_batching(predictions_one_hot_callable)

def get_inference_function(model: Callable, operator: Optional[Callable] = None):
    """
    Define the inference function according to the model type
    Parameters
    ----------
    model
        Model used for computing explanations.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    Returns
    -------
    inference_function
        Same definition as the operator.
    batch_inference_function
        An inference function which treat inputs and targets by batch,
        it has an additionnal parameter `batch_size`.
    """
    if operator is not None:
        # user specified a custom operator, we check if the operator is valid
        # and we wrap it to generate a batching version of this operator
        check_operator(operator)
        inference_function = operator
        batch_inference_function = operator_batching(operator)

    elif isinstance(model, tf.keras.Model):
        # no custom operator, for keras model we can backprop through the model
        inference_function = predictions_operator
        batch_inference_function = batch_predictions

    elif isinstance(model, (tf.Module, tf.keras.layers.Layer)):
        # maybe a custom model (e.g. tf-lite), we can't backprop through it
        inference_function = predictions_operator
        batch_inference_function = batch_predictions

    else:
        # completely unknown model (e.g. sklearn), we can't backprop through it
        inference_function = predictions_one_hot_callable
        batch_inference_function = batch_predictions_one_hot_callable

    return inference_function, batch_inference_function
