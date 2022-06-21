import numpy as np
import pytest
import rescaling


def setup_scaling(tag, y_l, y_u, scaled_y):
    p = rescaling.Params(y_l, y_u)
    r = rescaling.from_tag(tag, p_max=rescaling.Params(y_l=Y_MIN, y_u=Y_MAX, x_l=p[0].x_l, x_u=p[0].x_u))

    if p.B is None:
        x = np.linspace(p.x_l, p.x_u, N)
        y = np.linspace(p.y_l, p.y_u, N)
    else:
        t = np.reshape(np.linspace(0, 1, p.B), (p.B, 1))
        x = (1 - t) * p.x_l + t * p.x_u
        y = (1 - t) * p.y_l + t * p.y_u

    if scaled_y:
        y = r.rescale(x, p)
    return r, p, x, y


def check_scaling(r, p, x, y=None, y_true=None, x_true=None, dy_true=None, dx_true=None):
    if y_true is not None:
        ys = r.rescale(x, p)
        assert all_close(ys, y_true)

    if dy_true is not None:
        dys = r.grad(x, p)
        assert all_close(dys, dy_true)

    if x_true is not None:
        xs = r.inverse(y, p)
        assert all_close(xs, x_true)

    if dx_true is not None:
        dxs = r.inverse_grad(y, p)
        assert all_close(dxs, dx_true)


def check_grad(x, y, dy):
    for d in range(x.shape[1]):
        dy_fd = np.gradient(y[:,d], x[:,d])
        I = np.concatenate([[False], ~np.isnan(dy_fd[1:-1]), [False]])  # Skip nans and first and last element
        assert all_close(dy[I,d], dy_fd[I], mse_tol=5e0)  # rtol=1e-2, atol=1e-2 was too restrictive near discrete gradient jumps


def all_close(a, b, rtol=1e-5, atol=1e-8, mse_tol=None):
    if mse_tol is None:
        e = np.abs(a - b) - atol - rtol * np.abs(b)
        condition = np.allclose(a, b, rtol=rtol, atol=atol)
    else:
        e = (a - b)**2
        condition = np.mean((a - b)**2) < mse_tol

    if condition:
        return True
    else:
        i = np.argmax(e)
        print(f"All close failed with maximum tolerance error {e[i]}: a[{i}]={a[i]} != b[{i}]={b[i]}")
        import matplotlib.pyplot as plt
        plt.plot(a)
        plt.plot(b)
        plt.show()
        return False


N = 501
EPS = rescaling.Params.EPS
BOUNDS = [
    (-5, 5),  # Symmetric; scalar dim
    ([-1, -10], [10, 1]),  # Asymmetric; vector dim
    ([1, -10], [10, -1]),  # Forced asymmetric; vector dim
]
BATCH_BOUNDS = [
    (np.reshape(np.linspace(1, -5, 7), (-1, 1)), np.reshape(np.linspace(5, -1, 7), (-1, 1))),  # Combined ; batched scalar dim
    (np.reshape(np.linspace([-5, 1], [5, -1], 21), (-1, 2)), np.reshape(np.linspace([5, 5], [10, 1], 21), (-1, 2))),  # ? ; batched vector dim
]
Y_MIN, Y_MAX = -10, 10
TAGS = ["lin", "pwl", "hyp", "pwq", "clip"]


def test_rescale_params():
    p = rescaling.Params(-5, 5)
    assert p.y_l == -5
    assert p.y_u == 5
    assert p.y_0 == 0

    # Altered (forced) rescaling with y_0 outside [y_l;y_u]
    p = rescaling.Params(1, 10)
    assert p.y_l == 1
    assert p.y_u == 10
    assert p.y_0 == p.y_l + EPS

    p = rescaling.Params(-10, -1)
    assert p.y_l == -10
    assert p.y_u == -1
    assert p.y_0 == p.y_u - EPS

    p = rescaling.Params(10, 10)
    assert p.y_l == 10
    assert p.y_u == 10
    assert p.y_0 == 10

    # Altered (forced) rescaling with y_0 within [y_l;y_u]
    p = rescaling.Params(-EPS/2, 10)
    assert p.y_l == -EPS/2
    assert p.y_u == 10
    assert p.y_0 == EPS/2

    p = rescaling.Params(-10, EPS/2)
    assert p.y_l == -10
    assert p.y_u == EPS/2
    assert p.y_0 == -EPS/2

    p = rescaling.Params(-EPS/4, EPS/2)
    assert p.y_l == -EPS/4
    assert p.y_u == EPS/2
    assert p.y_0 == EPS/8

    # Check inverse
    p = rescaling.Params(1, 2, 3)
    p_inv = p.inverse
    assert p.y_l == p_inv.x_l and p.y_0 == p_inv.x_0 and p.y_u == p_inv.x_u
    assert p.x_l == p_inv.y_l and p.x_0 == p_inv.y_0 and p.x_u == p_inv.y_u

    # Check indexing
    p = rescaling.Params([[1, 2], [3, 4]], 10)
    assert np.all(p[1].y_l == p.y_l[1])
    assert np.all(p[1,1].y_l == p.y_l[1,1])
    assert np.all(p[:,1].y_l == p.y_l[:,1])
    assert np.all(p[1,:].y_l == p.y_l[1,:])

    p = rescaling.Params([[1], [3]], 10)
    assert np.all(p[1].y_l == p.y_l[1])
    assert np.all(p[1,0].y_l == p.y_l[1,0])
    assert np.all(p[:,0].y_l == p.y_l[:,0])
    assert np.all(p[1,:].y_l == p.y_l[1,:])

    p = rescaling.Params([1, 2], 10)
    assert np.all(p[1].y_l == p.y_l)
    assert np.all(p[1,0].y_l == p.y_l[0])
    assert np.all(p[:,0].y_l == p.y_l[0])
    assert np.all(p[1,:].y_l == p.y_l)

    # Invalid rescaling parameters
    with pytest.raises(AssertionError):
        rescaling.Params(1, -1)


@pytest.mark.parametrize(["y_l", "y_u"], [*BOUNDS, *BATCH_BOUNDS])
@pytest.mark.parametrize("tag", TAGS)
def test_rescaling_batch(y_l, y_u, tag):
    r, p, x, y = setup_scaling(tag, y_l, y_u, scaled_y=False)

    x_s, dx_s = np.empty(y.shape), np.empty(y.shape)
    y_s, dy_s = np.empty(x.shape), np.empty(x.shape)

    # Single input
    for i, xi in enumerate(x):
        pi = p if p.B is None else p[i]
        y_s[i, :] = r.rescale(xi, pi)
        dy_s[i, :] = r.grad(xi, pi)

    for i, yi in enumerate(y):
        pi = p if p.B is None else p[i]
        x_s[i, :] = r.inverse(yi, pi)
        dx_s[i, :] = r.inverse_grad(yi, pi)

    # Batched input
    y_m = r.rescale(x, p)
    dy_m = r.grad(x, p)
    x_m = r.inverse(y, p)
    dx_m = r.inverse_grad(y, p)

    assert np.allclose(y_m, y_s)
    assert np.allclose(dy_m, dy_s)
    assert np.allclose(x_m, x_s)
    assert np.allclose(dx_m, dx_s)


@pytest.mark.parametrize(["y_l", "y_u"], BOUNDS)
@pytest.mark.parametrize("tag", TAGS)
def test_rescaling_inverse(y_l, y_u, tag):
    r, p, x, y = setup_scaling(tag, y_l, y_u, scaled_y=True)
    assert p.B is None

    if r.INVERTIBLE:
        x_inv = r.inverse(y, p)
        assert np.allclose(x_inv, x)
    else:
        pytest.skip()  # Test only works for fully invertible functions


# @pytest.mark.skip("Too many false positives around discrete jumps with current checks")  # TODO: change correctness conditions
@pytest.mark.parametrize(["y_l", "y_u"], BOUNDS)
@pytest.mark.parametrize("tag", TAGS)
def test_rescaling_grad(y_l, y_u, tag):
    r, p, x, y = setup_scaling(tag, y_l, y_u, scaled_y=True)
    assert p.B is None

    ys = r.rescale(x, p)
    dy = r.grad(x, p)
    check_grad(x, ys, dy)

    xs = r.inverse(y, p)
    dx = r.inverse_grad(y, p)
    check_grad(y, xs, dx)


# Specific tests for rescaling correctness and corner cases
@pytest.mark.parametrize(["y_l", "y_u"], BOUNDS)
def test_lin(y_l, y_u):
    r, p, x, y = setup_scaling("lin", y_l, y_u, scaled_y=True)

    t = np.reshape(np.linspace(0, 1, N), (N, 1))
    y_true = (1-t)*p.y_l + t*p.y_u
    x_true = (1-t)*p.x_l + t*p.x_u
    dy_true = (p.y_u - p.y_l) / (p.x_u - p.x_l) * np.ones((N, 1))
    dx_true = 1 / dy_true
    check_scaling(r, p, x, y, y_true, x_true, dy_true, dx_true)


def test_lin_eq():
    r, p, x, y = setup_scaling("lin", 10, 10, scaled_y=True)
    assert (p.y_0 == p.y_l) and (p.y_0 == p.y_u)

    y_true = np.full((N, p.DIM), p.y_0)
    x_true = np.full((N, p.DIM), p.x_0)
    dy_true = np.zeros((N, p.DIM))
    dx_true = np.zeros((N, p.DIM))
    check_scaling(r, p, x, y, y_true, x_true, dy_true, dx_true)


@pytest.mark.parametrize(["y_l", "y_u"], BOUNDS)
def test_pwl(y_l, y_u):
    r, p, x, y = setup_scaling("pwl", y_l, y_u, scaled_y=True)

    t = np.reshape(np.linspace(0, 1, int(1+(N-1)/2)), (-1, 1))
    z = np.zeros((int((N-1)/2), 1))
    t_l = np.concatenate([1-t, z], axis=0)
    t_0 = np.concatenate([t[:-1], 1-t], axis=0)
    t_u = np.concatenate([z, t], axis=0)
    y_true = t_l*p.y_l + t_0*p.y_0 + t_u*p.y_u
    x_true = t_l*p.x_l + t_0*p.x_0 + t_u*p.x_u

    o = 1+z
    t_l = np.concatenate([o, [[0.5]], z], axis=0)
    t_u = np.concatenate([z, [[0.5]], o], axis=0)
    dyL = (p.y_l - p.y_0) / (p.x_l - p.x_0)
    dyU = (p.y_u - p.y_0) / (p.x_u - p.x_0)
    dy_true = t_l*dyL + t_u*dyU
    dx_true = t_l/dyL + t_u/dyU
    check_scaling(r, p, x, y, y_true, x_true, dy_true, dx_true)


def test_pwl_eq():
    r, p, x, y = setup_scaling("pwl", 10, 10, scaled_y=True)

    y_true = np.full((N, p.DIM), p.y_0)
    x_true = np.full((N, p.DIM), p.x_0)
    dy_true = np.zeros((N, p.DIM))
    dx_true = np.zeros((N, p.DIM))
    check_scaling(r, p, x, y, y_true, x_true, dy_true, dx_true)


def test_hyp_eq():
    r, p, x, y = setup_scaling("hyp", 10, 10, scaled_y=True)

    y_true = np.full((N, p.DIM), p.y_0)
    x_true = np.full((N, p.DIM), p.x_0)
    dy_true = np.zeros((N, p.DIM))
    dx_true = np.zeros((N, p.DIM))
    check_scaling(r, p, x, y, y_true, x_true, dy_true, dx_true)


def test_pwq_eq():
    r, p, x, y = setup_scaling("pwq", 10, 10, scaled_y=True)

    y_true = np.full((N, p.DIM), p.y_0)
    x_true = np.full((N, p.DIM), p.x_0)
    dy_true = np.zeros((N, p.DIM))
    dx_true = np.infty*np.ones((N, p.DIM))
    check_scaling(r, p, x, y, y_true, x_true, dy_true, dx_true)


def test_clip_eq():
    r, p, x, y = setup_scaling("clip", 10, 10, scaled_y=True)

    y_true = np.full((N, p.DIM), p.y_0)
    x_true = np.ones((N, p.DIM))
    dy_true = np.zeros((N, p.DIM))
    dx_true = np.zeros((N, p.DIM))
    check_scaling(r, p, x, y, y_true, x_true, dy_true, dx_true)


# TODO: add correctness and edge case tests for hyp, pwq and clip
