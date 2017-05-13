import random

import pytest

from learning import rlearn

#######################
# RLTable initial table
#######################
def test_rltable_initial_table():
    rl = rlearn.RLTable([0, 1], [0, 1])
    assert rl._reward_table == {
        0: {0: rl._initial_reward, 1: rl._initial_reward},
        1: {0: rl._initial_reward, 1: rl._initial_reward}
    }

#########################
# RLTable initial reward
#########################
def test_rltable_initial_reward():
    initial_reward = random.uniform(-1, 1)

    # Initial table
    rl = rlearn.RLTable([0], [0], initial_reward=initial_reward)
    assert rl._reward_table == {0: {0: initial_reward}}

    # Add action
    rl.add_action(1, 1)
    assert rl._reward_table == {0: {0: initial_reward}, 1: {1: initial_reward}}

#######################
# RLTable.get_action
#######################
def test_rltable_get_action():
    rl = rlearn.RLTable([0], [0, 1])
    rl._reward_table[0][1] = 999999999
    assert rl.get_action(0) == 1

    rl._reward_table[0][1] = -999999999
    assert rl.get_action(0) == 0

#######################
# RLTable.update_action
#######################
def test_rltable_update():
    rl = rlearn.RLTable([0, 1], [0, 1], initial_reward=1.0, update_rate=0.5)
    rl.update(0, 0, 0.0)
    assert rl._reward_table[0][0] == 0.5
    assert rl._reward_table[0][1] == 1.0
    assert rl._reward_table[1][0] == 1.0
    assert rl._reward_table[1][1] == 1.0

    rl.update(0, 1, 3.0)
    assert rl._reward_table[0][0] == 0.5
    assert rl._reward_table[0][1] == 2.0
    assert rl._reward_table[1][0] == 1.0
    assert rl._reward_table[1][1] == 1.0

    rl.update(1, 0, -1.0)
    assert rl._reward_table[0][0] == 0.5
    assert rl._reward_table[0][1] == 2.0
    assert rl._reward_table[1][0] == 0.0
    assert rl._reward_table[1][1] == 1.0

    # Different rate
    rl = rlearn.RLTable([0], [0], initial_reward=1.0, update_rate=1.0)
    rl.update(0, 0, 0.0)
    assert rl._reward_table[0][0] == 0.0

#########################
# RLTable._increment_all
#########################
def test_rltable_update_with_reward_growth():
    rl = rlearn.RLTable([0, 1], [0, 1], initial_reward=1.0,
                        update_rate=1.0, reward_growth=0.1)
    rl.update(0, 0, 0.0)
    assert rl._reward_table[0][0] == 0.1
    assert rl._reward_table[0][1] == 1.1
    assert rl._reward_table[1][0] == 1.1
    assert rl._reward_table[1][1] == 1.1

    rl = rlearn.RLTable([0, 1], [0, 1], initial_reward=1.0,
                        update_rate=1.0, reward_growth=-0.1)
    rl.update(0, 0, 0.0)
    assert rl._reward_table[0][0] == -0.1
    assert rl._reward_table[0][1] == 0.9
    assert rl._reward_table[1][0] == 0.9
    assert rl._reward_table[1][1] == 0.9

def test_rltable_increment_all():
    rl = rlearn.RLTable([0, 1], [0, 1], initial_reward=1.0)
    rl._increment_all(0.1)
    assert rl._reward_table[0][0] == 1.1
    assert rl._reward_table[0][1] == 1.1
    assert rl._reward_table[1][0] == 1.1
    assert rl._reward_table[1][1] == 1.1

    rl._increment_all(-0.2)
    assert (rl._reward_table[0][0] - 0.9) < 0.00000001
    assert (rl._reward_table[0][1] - 0.9) < 0.00000001
    assert (rl._reward_table[1][0] - 0.9) < 0.00000001
    assert (rl._reward_table[1][1] - 0.9) < 0.00000001

#######################
# RLTable.add_action
#######################
def test_rltable_add_action():
    rl = rlearn.RLTable([], [])
    assert rl._reward_table == {}

    # New state, new action
    rl.add_action(0, 0)
    assert rl._reward_table == {0: {0: rl._initial_reward}}

    # Old state, new action
    rl.add_action(0, 1)
    assert rl._reward_table == {0: {0: rl._initial_reward, 1: rl._initial_reward}}

    # New state, new action, after existing state added
    rl.add_action(1, 0)
    assert rl._reward_table == {
        0: {0: rl._initial_reward, 1: rl._initial_reward},
        1: {0: rl._initial_reward}
    }

def test_rltable_add_action_existing():
    rl = rlearn.RLTable([], [])
    assert rl._reward_table == {}

    # New state, new action
    rl.add_action(0, 0)
    assert rl._reward_table == {0: {0: rl._initial_reward}}

    # Old state, new action
    with pytest.raises(ValueError):
        rl.add_action(0, 0)
    assert rl._reward_table == {0: {0: rl._initial_reward}}

#######################
# RLTable.delete_action
#######################
def test_rltable_delete_action():
    rl = rlearn.RLTable([], [])
    assert rl._reward_table == {}

    # New state, new action
    rl.add_action(0, 0)
    assert rl._reward_table == {0: {0: rl._initial_reward}}

    # Old state, new action
    rl.add_action(0, 1)
    assert rl._reward_table == {0: {0: rl._initial_reward, 1: rl._initial_reward}}

    # Remove one action
    rl.delete_action(0, 1)
    assert rl._reward_table == {0: {0: rl._initial_reward}}

def test_rltable_delete_only_action_for_state():
    rl = rlearn.RLTable([], [])
    assert rl._reward_table == {}

    # New state, new action
    rl.add_action(0, 0)
    assert rl._reward_table == {0: {0: rl._initial_reward}}

    # Remove it
    rl.delete_action(0, 0)
    assert rl._reward_table == {}

def test_rltable_delete_action_that_doesnt_exist():
    rl = rlearn.RLTable([], [])
    assert rl._reward_table == {}

    # New state, new action
    rl.add_action(0, 0)
    assert rl._reward_table == {0: {0: rl._initial_reward}}

    # Delete non-existant action
    with pytest.raises(KeyError):
        rl.delete_action(0, 1)
    assert rl._reward_table == {0: {0: rl._initial_reward}}

    # Delete non-existant state
    with pytest.raises(KeyError):
        rl.delete_action(1, 0)
    assert rl._reward_table == {0: {0: rl._initial_reward}}
