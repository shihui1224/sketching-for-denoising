def grad_descent_fun(fun, x0, nb_it, alpha):
    x = x0
    f_val = []
    for i in range(nb_it):
        f, g = fun(x)
        x_new = x - alpha*g
        x = x_new
        f_val.append(f)
    return x, f_val
