"""Functions exposed directly in the garage namespace."""
from collections import defaultdict

from dowel import tabular
import numpy as np

from garage import EpisodeBatch, StepType
from garage.misc.tensor_utils import discount_cumsum


class _Default:  # pylint: disable=too-few-public-methods
    """A wrapper class to represent default arguments.

    Args:
        val (object): Argument value.

    """

    def __init__(self, val):
        self.val = val


def make_optimizer(optimizer_type, module=None, **kwargs):
    """Create an optimizer for pyTorch & tensorflow algos.

    Args:
        optimizer_type (Union[type, tuple[type, dict]]): Type of optimizer.
            This can be an optimizer type such as 'torch.optim.Adam' or a
            tuple of type and dictionary, where dictionary contains arguments
            to initialize the optimizer e.g. (torch.optim.Adam, {'lr' : 1e-3})
        module (optional): If the optimizer type is a `torch.optimizer`.
            The `torch.nn.Module` module whose parameters needs to be optimized
            must be specify.
        kwargs (dict): Other keyword arguments to initialize optimizer. This
            is not used when `optimizer_type` is tuple.

    Returns:
        torch.optim.Optimizer: Constructed optimizer.

    Raises:
        ValueError: Raises value error when `optimizer_type` is tuple, and
            non-default argument is passed in `kwargs`.

    """
    if isinstance(optimizer_type, tuple):
        opt_type, opt_args = optimizer_type
        for name, arg in kwargs.items():
            if not isinstance(arg, _Default):
                raise ValueError('Should not specify {} and explicit \
                    optimizer args at the same time'.format(name))
        if module is not None:
            return opt_type(module.parameters(), **opt_args)
        else:
            return opt_type(**opt_args)

    opt_args = {
        k: v.val if isinstance(v, _Default) else v
        for k, v in kwargs.items()
    }
    if module is not None:
        return optimizer_type(module.parameters(), **opt_args)
    else:
        return optimizer_type(**opt_args)


def log_multitask_performance(itr, batch, discount, name_map=None):
    r"""Log performance of episodes from multiple tasks.

    Args:
        itr (int): Iteration number to be logged.
        batch (EpisodeBatch): Batch of episodes. The episodes should have
            either the "task_name" or "task_id" `env_infos`. If the "task_name"
            is not present, then `name_map` is required, and should map from
            task id's to task names.
        discount (float): Discount used in computing returns.
        name_map (dict[int, str] or None): Mapping from task id's to task
            names. Optional if the "task_name" environment info is present.
            Note that if provided, all tasks listed in this map will be logged,
            even if there are no episodes present for them.

    Returns:
        numpy.ndarray: Undiscounted returns averaged across all tasks. Has
            shape :math:`(N \bullet [T])`.

    """
    eps_by_name = defaultdict(list)
    for eps in batch.split():
        task_name = '__unnamed_task__'
        if 'task_name' in eps.env_infos:
            task_name = eps.env_infos['task_name'][0]
        elif 'task_id' in eps.env_infos:
            name_map = {} if name_map is None else name_map
            task_id = eps.env_infos['task_id'][0]
            task_name = name_map.get(task_id, 'Task #{}'.format(task_id))
        eps_by_name[task_name].append(eps)
    if name_map is None:
        task_names = eps_by_name.keys()
    else:
        task_names = name_map.values()
    for task_name in task_names:
        if task_name in eps_by_name:
            episodes = eps_by_name[task_name]
            log_performance(itr,
                            EpisodeBatch.concatenate(*episodes),
                            discount,
                            prefix=task_name)
        else:
            with tabular.prefix(task_name + '/'):
                tabular.record('Iteration', itr)
                tabular.record('NumEpisodes', 0)
                tabular.record('AverageDiscountedReturn', np.nan)
                tabular.record('AverageReturn', np.nan)
                tabular.record('StdReturn', np.nan)
                tabular.record('MaxReturn', np.nan)
                tabular.record('MinReturn', np.nan)
                tabular.record('TerminationRate', np.nan)
                tabular.record('SuccessRate', np.nan)

    return log_performance(itr, batch, discount=discount, prefix='Average')


def log_performance(itr, batch, discount, prefix='Evaluation'):
    """Evaluate the performance of an algorithm on a batch of episodes.

    Args:
        itr (int): Iteration number.
        batch (EpisodeBatch): The episodes to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    returns = []
    undiscounted_returns = []
    termination = []
    success = []
    rewards = []
    grasp_success = []
    near_object = []
    episode_mean_grasp_reward = []
    episode_max_grasp_reward = []
    episode_min_grasp_reward = []
    episode_mean_in_place_reward = []
    episode_max_in_place_reward = []
    episode_min_in_place_reward = []
    for eps in batch.split():
        rewards.append(eps.rewards)
        returns.append(discount_cumsum(eps.rewards, discount))
        undiscounted_returns.append(sum(eps.rewards))
        termination.append(
            float(
                any(step_type == StepType.TERMINAL
                    for step_type in eps.step_types)))
        if 'success' in eps.env_infos:
            success.append(float(eps.env_infos['success'].any()))
        if 'grasp_success' in eps.env_infos:
            grasp_success.append(float(eps.env_infos['grasp_success'].any()))
        if 'near_object' in eps.env_infos:
            near_object.append(float(eps.env_infos['near_object'].any()))
        if 'grasp_reward' in eps.env_infos:
            episode_mean_grasp_reward.append(
                np.mean(eps.env_infos['grasp_reward']))
            episode_max_grasp_reward.append(max(eps.env_infos['grasp_reward']))
            episode_min_grasp_reward.append(min(eps.env_infos['grasp_reward']))
        if 'in_place_reward' in eps.env_infos:
            episode_mean_in_place_reward.append(
                np.mean(eps.env_infos['in_place_reward']))
            episode_max_in_place_reward.append(
                max(eps.env_infos['in_place_reward']))
            episode_min_in_place_reward.append(
                min(eps.env_infos['in_place_reward']))

    average_discounted_return = np.mean([rtn[0] for rtn in returns])

    with tabular.prefix(prefix + '/'):
        tabular.record('Iteration', itr)
        tabular.record('NumEpisodes', len(returns))
        tabular.record('MinReward', np.min(rewards))
        tabular.record('MaxReward', np.max(rewards))
        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))
        tabular.record('TerminationRate', np.mean(termination))
        if success:
            tabular.record('SuccessRate', np.mean(success))
        if grasp_success:
            tabular.record('GraspSuccessRate', np.mean(grasp_success))
        if near_object:
            tabular.record('NearObject', np.mean(near_object))
        if episode_mean_grasp_reward:
            tabular.record('EpisodeMeanGraspReward',
                           np.mean(episode_mean_grasp_reward))
            tabular.record('EpisodeMeanMaxGraspReward',
                           np.mean(episode_max_grasp_reward))
            tabular.record('EpisodeMeanMinGraspReward',
                           np.mean(episode_min_grasp_reward))
        if episode_mean_in_place_reward:
            tabular.record('EpisodeMeanInPlaceReward',
                           np.mean(episode_mean_in_place_reward))
            tabular.record('EpisodeMeanMaxInPlaceReward',
                           np.mean(episode_max_in_place_reward))
            tabular.record('EpisodeMeanMinInPlaceReward',
                           np.mean(episode_min_in_place_reward))

    return undiscounted_returns
