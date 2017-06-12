from learning import optimize

# NOTE: Not all combinations are tested
##################################
# Problem._get_obj
##################################
def test_optimizer_get_obj_obj_func():
    problem = optimize.Problem(obj_func=lambda x: x)
    assert problem.get_obj(1) == 1

def test_optimizer_get_obj_obj_jac_func():
    problem = optimize.Problem(obj_jac_func=lambda x: (x, x+1))
    assert problem.get_obj(1) == 1

def test_optimizer_get_obj_obj_jac_hess_func():
    problem = optimize.Problem(obj_jac_hess_func=lambda x: (x, x+1, x+2))
    assert problem.get_obj(1) == 1

##################################
# Problem._get_jac
##################################
def test_optimizer_get_jac_jac_func():
    problem = optimize.Problem(jac_func=lambda x: x+1)
    assert problem.get_jac(1) == 2

def test_optimizer_get_jac_obj_jac_func():
    problem = optimize.Problem(obj_jac_func=lambda x: (x, x+1))
    assert problem.get_jac(1) == 2

##################################
# Problem._get_hess
##################################
def test_optimizer_get_hess_hess_func():
    problem = optimize.Problem(hess_func=lambda x: x+2)
    assert problem.get_hess(1) == 3

def test_optimizer_get_hess_obj_hess_func():
    problem = optimize.Problem(obj_hess_func=lambda x: (x, x+2))
    assert problem.get_hess(1) == 3

##################################
# Problem._get_obj_jac
##################################
def test_optimizer_get_obj_jac_obj_jac_func():
    problem = optimize.Problem(obj_jac_func=lambda x: (x, x+1))
    assert problem.get_obj_jac(1) == (1, 2)

def test_optimizer_get_obj_jac_obj_jac_hess_func():
    problem = optimize.Problem(obj_jac_hess_func=lambda x: (x, x+1, x+2))
    assert tuple(problem.get_obj_jac(1)) == (1, 2)

def test_optimizer_get_obj_jac_individual_obj_jac():
    problem = optimize.Problem(obj_func=lambda x: x, jac_func=lambda x: x+1)
    assert tuple(problem.get_obj_jac(1)) == (1, 2)

##################################
# Problem._get_obj_jac_hess
##################################
def test_optimizer_get_obj_jac_hess_obj_jac_hess_func():
    problem = optimize.Problem(obj_jac_hess_func=lambda x: (x, x+1, x+2))
    assert tuple(problem.get_obj_jac_hess(1)) == (1, 2, 3)

def test_optimizer_get_obj_jac_hess_obj_jac_func_individual_hess():
    problem = optimize.Problem(obj_jac_func=lambda x: (x, x+1), hess_func=lambda x: x+2)
    assert tuple(problem.get_obj_jac_hess(1)) == (1, 2, 3)

def test_optimizer_get_obj_jac_hess_obj_hess_func_individual_jac():
    problem = optimize.Problem(obj_hess_func=lambda x: (x, x+2), jac_func=lambda x: x+1)
    assert tuple(problem.get_obj_jac_hess(1)) == (1, 2, 3)

def test_optimizer_get_obj_jac_hess_individual_obj_jac_hess():
    problem = optimize.Problem(obj_func=lambda x: x, jac_func=lambda x: x+1,
                                   hess_func=lambda x: x+2)
    assert tuple(problem.get_obj_jac_hess(1)) == (1, 2, 3)
