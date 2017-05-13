"""Reinforcement learning models."""
import operator

class RLTable(object):
    """Reinforcement learning using a (state, action) -> reward table.

    No generalization.
    No delayed reward.

    Each (state, action) pair learns its own reward value, independent
    of other (state, action) pairs.
    """
    def __init__(self, states, actions, initial_reward=2.0, update_rate=0.5,
                 reward_growth=0.0):
        if update_rate <= 0.0 or update_rate > 1.0:
            raise ValueError('update_rate must be within (0, 1]')

        self._initial_reward = initial_reward
        self._update_rate = update_rate
        self._reward_growth = reward_growth

        # Make initial table, for each state, action pair
        self._reward_table = {}
        for state in states:
            for action in actions:
                self.add_action(state, action)

    def get_action(self, state):
        """Return best action for this state.

        Return action for state with largest reward.
        """
        # TODO: Optionally stochastically select action, weighted by expected reward
        return max(self._reward_table[state].iteritems(), key=operator.itemgetter(1))[0]

    def update(self, state, action, new_reward):
        """Update reward for given (state, action)."""
        self._reward_table[state][action] = _adjust_value(
            self._reward_table[state][action], new_reward, self._update_rate)

        # Update all reward values by self._reward_growth
        if self._reward_growth != 0.0:
            self._increment_all(self._reward_growth)

    def _increment_all(self, increment):
        for state in self._reward_table:
            for action in self._reward_table[state]:
                self._reward_table[state][action] += increment

    def add_action(self, state, action):
        """Add new action to track."""
        # Get action table for given state
        try:
            state_table = self._reward_table[state]
        except KeyError:
            state_table = {}
            self._reward_table[state] = state_table

        # Add action for this state, if it does not exist
        if action in state_table:
            raise ValueError('(state, action) pair already exists')
        else:
            state_table[action] = self._initial_reward

    def delete_action(self, state, action):
        """Remove action from tracking."""
        self._reward_table[state].pop(action)
        if self._reward_table[state] == {}:
            self._reward_table.pop(state)

def _adjust_value(old_value, new_value, rate):
    """Return old_value, incrementally adjusted towards new_value.

    How much it is adjusted is determined by rate.
    """
    return old_value + rate*(new_value - old_value)
