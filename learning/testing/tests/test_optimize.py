from learning import optimize

##################################
# Optimizer._get_obj
##################################
def test_optimizer_get_obj_obj_func():
    optimizer = optimize.Optimizer(obj_func=lambda x: x)
    assert optimizer._get_obj(1) == 1

def test_optimizer_get_obj_obj_jac_func():
    optimizer = optimize.Optimizer(obj_jac_func=lambda x: (x, x+1))
    assert optimizer._get_obj(1) == 1

def test_optimizer_get_obj_obj_jac_hess_func():
    optimizer = optimize.Optimizer(obj_jac_hess_func=lambda x: (x, x+1, x+2))
    assert optimizer._get_obj(1) == 1

##################################
# Optimizer._get_jac
##################################
def test_optimizer_get_jac_jac_func():
    optimizer = optimize.Optimizer(jac_func=lambda x: x+1)
    assert optimizer._get_jac(1) == 2

def test_optimizer_get_jac_obj_jac_func():
    optimizer = optimize.Optimizer(obj_jac_func=lambda x: (x, x+1))
    assert optimizer._get_jac(1) == 2

##################################
# Optimizer._get_hess
##################################
def test_optimizer_get_hess_hess_func():
    optimizer = optimize.Optimizer(hess_func=lambda x: x+2)
    assert optimizer._get_hess(1) == 3

def test_optimizer_get_hess_obj_hess_func():
    optimizer = optimize.Optimizer(obj_hess_func=lambda x: (x, x+2))
    assert optimizer._get_hess(1) == 3

##################################
# Optimizer._get_obj_jac
##################################
def test_optimizer_get_obj_jac_obj_jac_func():
    optimizer = optimize.Optimizer(obj_jac_func=lambda x: (x, x+1))
    assert optimizer._get_obj_jac(1) == (1, 2)

def test_optimizer_get_obj_jac_obj_jac_hess_func():
    optimizer = optimize.Optimizer(obj_jac_hess_func=lambda x: (x, x+1, x+2))
    assert tuple(optimizer._get_obj_jac(1)) == (1, 2)

def test_optimizer_get_obj_jac_individual_obj_jac():
    optimizer = optimize.Optimizer(obj_func=lambda x: x, jac_func=lambda x: x+1)
    assert tuple(optimizer._get_obj_jac(1)) == (1, 2)

##################################
# Optimizer._get_obj_jac_hess
##################################
def test_optimizer_get_obj_jac_hess_obj_jac_hess_func():
    optimizer = optimize.Optimizer(obj_jac_hess_func=lambda x: (x, x+1, x+2))
    assert tuple(optimizer._get_obj_jac_hess(1)) == (1, 2, 3)

def test_optimizer_get_obj_jac_hess_obj_jac_func_individual_hess():
    optimizer = optimize.Optimizer(obj_jac_func=lambda x: (x, x+1), hess_func=lambda x: x+2)
    assert tuple(optimizer._get_obj_jac_hess(1)) == (1, 2, 3)

def test_optimizer_get_obj_jac_hess_obj_hess_func_individual_jac():
    optimizer = optimize.Optimizer(obj_hess_func=lambda x: (x, x+2), jac_func=lambda x: x+1)
    assert tuple(optimizer._get_obj_jac_hess(1)) == (1, 2, 3)

def test_optimizer_get_obj_jac_hess_individual_obj_jac_hess():
    optimizer = optimize.Optimizer(obj_func=lambda x: x, jac_func=lambda x: x+1,
                                   hess_func=lambda x: x+2)
    assert tuple(optimizer._get_obj_jac_hess(1)) == (1, 2, 3)
