from typing import Union, List
from dataclasses import dataclass, fields
import numpy as np

T = Union[float, List, np.ndarray]


@dataclass
class Params:
    y_l: T          # Lower bound of output variable (y)
    y_u: T          # Upper bound of output variable (y)
    y_0: T = 0.0    # Output for x==x_0
    x_l: T = -1.0   # Lower bound of input variable (x)
    x_u: T = 1.0    # Upper bound of input variable (x)
    x_0: T = 0.0    # Anchor point (x)

    EPS = 1e-2      # Epsilon value used for altered (forced) rescaling

    def __post_init__(self):
        # Convert all fields to ndarray with same shape
        vals = [np.atleast_1d(getattr(self, field.name)) for field in fields(self)]
        shape = np.broadcast(*vals).shape
        dtype = np.result_type(*vals)
        for field, val in zip(fields(self), vals):
            setattr(self, field.name, np.full(shape, val, dtype=dtype))
        # Check validity of fields
        assert 0 < len(shape) <= 2
        assert np.all(self.x_l <= self.x_u)
        assert np.all(self.y_l <= self.y_u)
        self.x_0 = np.clip(self.x_0, self.x_l, self.x_u)
        self.y_0 = np.clip(self.y_0, self.y_l, self.y_u)
        self._update_anchor()  # Modify anchor point (x_0, y_0) for altered (forced) rescaling when necessary
        self.DIM = shape[-1]  # Dimension of considered variables (x, y)
        self.B = shape[0] if len(shape) == 2 else None  # Batch dimension

    def _update_anchor(self):
        # Update anchors (x_0 and y_0 parameters) for altered (forced) rescaling.
        # This ensures v_l + EPS <= v_0 <= v_u - EPS when possible and v_0 = (v_l + v_u) / 2 otherwise.
        for v in ("x", "y"):
            v_l, v_0, v_u = getattr(self, f"{v}_l"), getattr(self, f"{v}_0"), getattr(self, f"{v}_u")
            if not np.all((v_l + self.EPS <= v_0) & (v_0 <= v_u - self.EPS)):
                # Modify x_0/y_0 parameter for altered (forced) rescaling
                M_l = v_l > v_0 - self.EPS
                M_u = v_u < v_0 + self.EPS
                M_s = v_u - v_l < 2 * self.EPS
                v_0[M_l] = v_l[M_l] + self.EPS
                v_0[M_u] = v_u[M_u] - self.EPS
                v_0[M_s] = (v_l[M_s] + v_u[M_s]) / 2

    @property
    def inverse(self):
        # Inverse parameters for rescaling functions based on involutions
        return Params(y_l=self.x_l, y_u=self.x_u, y_0=self.x_0, x_l=self.y_l, x_u=self.y_u, x_0=self.y_0)

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            # Index over batch dimension by default when a single index (slice) is given
            b = index
            c = slice(None)
        elif isinstance(index, tuple) and len(index) == 2 and isinstance(index[0], (int, slice)) and isinstance(index[1], (int, slice)):
            # Index over both batch and component dimension
            b, c = index
        else:
            raise IndexError("Invalid index supplied.")

        if self.B is None:
            return Params(y_l=self.y_l[c], y_u=self.y_u[c], y_0=self.y_0[c], x_l=self.x_l[c], x_u=self.x_u[c], x_0=self.x_0[c])
        else:
            return Params(y_l=self.y_l[b,c], y_u=self.y_u[b,c], y_0=self.y_0[b,c], x_l=self.x_l[b,c], x_u=self.x_u[b,c], x_0=self.x_0[b,c])


class Rescaling:
    __classes = {}

    INVOLUTION = True  # Rescaling function is involution based
    INVERTIBLE = True  # Rescaling function is invertible *over the whole domain* (specified by the rescaling parameters)

    def __init_subclass__(cls, tag=None, **kwargs):
        """ Subclasses are automatically registered if they provide a unique tag. """
        if tag is not None:
            assert tag not in cls.__classes
            cls.tag = tag
            cls.__classes[tag] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def from_tag(cls, tag, *args, **kwargs) -> "Rescaling":
        return cls.__classes[tag](*args, **kwargs)

    def __init__(self, *args, **kwargs):
        pass

    def rescale(self, x: T, p: Params) -> T:
        """Rescale the input x using the rescaling parameters p."""
        raise NotImplementedError("Subclasses should implement the rescaling function here")

    def grad(self, x: T, p: Params) -> T:
        """Calculate the gradient of the rescaling function for the input x using the rescaling parameters p."""
        raise NotImplementedError("Subclasses should implement the gradient of the rescaling function here")

    def inverse(self, y: T, p: Params) -> T:
        """Apply the inverse rescaling function on the output y using the (original) rescaling parameters p."""
        raise NotImplementedError("Subclasses should implement the inverse rescaling function here")

    def inverse_grad(self, y: T, p: Params) -> T:
        """Calculate the gradient of the inverse rescaling function for the output y using the (original) rescaling
        parameters p."""
        raise NotImplementedError("Subclasses should implement the gradient of the inverse rescaling function here")


class Linear(Rescaling, tag="lin"):
    """Linear rescaling function (sigma_lin).
    Note: Does not take the anchor point (x_0, y_0) into account."""

    INVOLUTION = True
    INVERTIBLE = True

    def rescale(self, x: T, p: Params) -> T:
        x = np.reshape(np.clip(x, p.x_l, p.x_u), (-1, p.DIM))
        y = p.y_l + (p.y_u - p.y_l) / (p.x_u - p.x_l) * (x - p.x_l)
        return np.where(p.x_u != p.x_l, y, p.y_0)

    def grad(self, x: T, p: Params) -> T:
        x = np.reshape(np.clip(x, p.x_l, p.x_u), (-1, p.DIM))
        dy = (p.y_u - p.y_l) / (p.x_u - p.x_l) * np.ones(x.shape)  # Ensure correct dimension of gradient
        return np.where(p.x_u != p.x_l, dy, 0)

    def inverse(self, y: T, p: Params) -> T:
        return self.rescale(y, p.inverse)  # Involution based rescaling

    def inverse_grad(self, y: T, p: Params) -> T:
        return self.grad(y, p.inverse)  # Involution based rescaling


class PiecewiseLinear(Rescaling, tag="pwl"):
    """Piecewise linear rescaling function (sigma_pwl)."""

    INVOLUTION = True
    INVERTIBLE = True

    def pL(self, p: Params) -> Params:
        """Return linear rescaling params for lower piece."""
        return Params(y_l=p.y_l, y_u=p.y_0, x_l=p.x_l, x_u=p.x_0)

    def pU(self, p: Params) -> Params:
        """Return linear rescaling params for upper piece."""
        return Params(y_l=p.y_0, y_u=p.y_u, x_l=p.x_0, x_u=p.x_u)

    def rescale(self, x: T, p: Params) -> T:
        x = np.reshape(np.clip(x, p.x_l, p.x_u), (-1, p.DIM))
        yL = Linear().rescale(x, self.pL(p))  # p.y_0 + (p.y_l - p.y_0) * (x - p.x_0) / (p.x_l - p.x_0)
        yU = Linear().rescale(x, self.pU(p))  # p.y_0 + (p.y_u - p.y_0) * (x - p.x_0) / (p.x_u - p.x_0)
        return np.where(x < p.x_0, yL, np.where(x > p.x_0, yU, p.y_0))

    def grad(self, x: T, p: Params) -> T:
        x = np.reshape(np.clip(x, p.x_l, p.x_u), (-1, p.DIM))
        dyL = Linear().grad(x, self.pL(p))  # (p.y_l - p.y_0) / (p.x_l - p.x_0)
        dyU = Linear().grad(x, self.pU(p))  # (p.y_u - p.y_0) / (p.x_u - p.x_0)
        dy0 = (dyL + dyU) / 2
        return np.where(x < p.x_0, dyL, np.where(x > p.x_0, dyU, dy0))

    def inverse(self, y: T, p: Params) -> T:
        return self.rescale(y, p.inverse)  # Involution based rescaling

    def inverse_grad(self, y: T, p: Params) -> T:
        return self.grad(y, p.inverse)  # Involution based rescaling


class Hyperbolic(Rescaling, tag="hyp"):
    """Hyperbolic rescaling function (sigma_hyp)."""

    INVOLUTION = True
    INVERTIBLE = True

    def rescale(self, x: T, p: Params) -> T:
        x = np.reshape(np.clip(x, p.x_l, p.x_u), (-1, p.DIM))
        y = p.y_0 + ((p.y_l-p.y_0)*(p.y_u-p.y_0)*(p.x_u-p.x_l)*(x-p.x_0)) /\
            (((p.y_l - p.y_0)*(p.x_u - p.x_0) - (p.y_u - p.y_0)*(p.x_l - p.x_0))*(x - p.x_0) + (p.x_l - p.x_0)*(p.x_u - p.x_0)*(p.y_u - p.y_l))
        return np.where((p.x_u != p.x_l) & (p.y_u != p.y_l), y, p.y_0)

    def grad(self, x: T, p: Params) -> T:
        x = np.reshape(np.clip(x, p.x_l, p.x_u), (-1, p.DIM))
        dy = ((p.y_l - p.y_0)*(p.y_u - p.y_0)*(p.y_u - p.y_l)*(p.x_l - p.x_0)*(p.x_u - p.x_0)*(p.x_u - p.x_l)) / \
             (((p.y_l - p.y_0)*(p.x_u - p.x_0) - (p.y_u - p.y_0)*(p.x_l - p.x_0))*(x - p.x_0) + (p.x_l - p.x_0)*(p.x_u - p.x_0)*(p.y_u - p.y_l))**2
        return np.where((p.x_u != p.x_l) & (p.y_u != p.y_l), dy, 0)

    def inverse(self, y: T, p: Params) -> T:
        return self.rescale(y, p.inverse)  # Involution based rescaling

    def inverse_grad(self, y: T, p: Params) -> T:
        return self.grad(y, p.inverse)  # Involution based rescaling


class PiecewiseQuadratic(Rescaling, tag="pwq"):
    """Piecewise quadratic rescaling function.
    Note: This rescaling function is not involution based!"""

    INVOLUTION = False
    INVERTIBLE = True

    def m(self, p: Params) -> float:
        """Slope at anchor point."""
        # To have a monotonically increasing and invertable rescaling function within the rescaling domain and range,
        # m should always be less than 2*(y_l - y_0)/(x_l - x_0) and 2*(y_u - y_0)/(x_u - x_0)
        s = 0.5  # 0<s<1 gives an inflection point at (x_0, y_0) ; s=1 gives linear + quadratic piece ; 1<s<2 gives convex function (not verified/proven)
        return s*np.abs(np.minimum((p.y_l - p.y_0)/(p.x_l - p.x_0), (p.y_u - p.y_0)/(p.x_u - p.x_0)))

    def rescale(self, x: T, p: Params) -> T:
        x = np.reshape(np.clip(x, p.x_l, p.x_u), (-1, p.DIM))
        m = self.m(p)
        yL = p.y_0 - m / (p.x_l - p.x_0) * (x - p.x_0) * (x - p.x_l) + (p.y_l - p.y_0) * (x - p.x_0)**2 / (p.x_l - p.x_0)**2
        yU = p.y_0 - m / (p.x_u - p.x_0) * (x - p.x_0) * (x - p.x_u) + (p.y_u - p.y_0) * (x - p.x_0)**2 / (p.x_u - p.x_0)**2
        return np.where(x < p.x_0, yL, np.where(x > p.x_0, yU, p.y_0))

    def grad(self, x: T, p: Params) -> T:
        x = np.reshape(np.clip(x, p.x_l, p.x_u), (-1, p.DIM))
        m = self.m(p)
        dyL = - m / (p.x_l - p.x_0) * (2*x - p.x_0 - p.x_l) + 2 * (p.y_l - p.y_0) * (x - p.x_0) / (p.x_l - p.x_0)**2
        dyU = - m / (p.x_u - p.x_0) * (2*x - p.x_0 - p.x_u) + 2 * (p.y_u - p.y_0) * (x - p.x_0) / (p.x_u - p.x_0)**2
        return np.where(x < p.x_0, dyL, np.where(x > p.x_0, dyU, m))

    def inverse(self, y: T, p: Params) -> T:
        y = np.reshape(np.clip(y, p.y_l, p.y_u), (-1, p.DIM))
        m = self.m(p)
        mL = m * (p.x_l - p.x_0) / (p.y_l - p.y_0)
        mU = m * (p.x_u - p.x_0) / (p.y_u - p.y_0)
        err_cfg = np.seterr(invalid="ignore")  # Temporarily silence "invalid value encountered in sqrt" warning, invalid values are automatically dealt with by the where function
        xL = p.x_0 + (p.x_l - p.x_0) * (np.sqrt(mL*mL + 4*(1 - mL)*(y - p.y_0)/(p.y_l - p.y_0)) - mL) / (2 - 2*mL)
        xU = p.x_0 + (p.x_u - p.x_0) * (np.sqrt(mU*mU + 4*(1 - mU)*(y - p.y_0)/(p.y_u - p.y_0)) - mU) / (2 - 2*mU)
        np.seterr(**err_cfg)  # Restore old error configuration
        return np.where(y < p.y_0, xL, np.where(y > p.y_0, xU, p.x_0))

    def inverse_grad(self, y: T, p: Params) -> T:
        y = np.reshape(np.clip(y, p.y_l, p.y_u), (-1, p.DIM))
        m = self.m(p)
        mL = m * (p.x_l - p.x_0) / (p.y_l - p.y_0)
        mU = m * (p.x_u - p.x_0) / (p.y_u - p.y_0)
        err_cfg = np.seterr(invalid="ignore")  # Temporarily silence "invalid value encountered in sqrt" warning, invalid values are automatically dealt with by the where function
        dxL = (p.x_l - p.x_0) / (p.y_l - p.y_0) / np.sqrt(mL*mL + 4*(1 - mL)*(y - p.y_0)/(p.y_l - p.y_0))
        dxU = (p.x_u - p.x_0) / (p.y_u - p.y_0) / np.sqrt(mU*mU + 4*(1 - mU)*(y - p.y_0)/(p.y_u - p.y_0))
        np.seterr(**err_cfg)  # Restore old error configuration
        return np.where(y < p.y_0, dxL, np.where(y > p.y_0, dxU, 1/m))


class Clipping(Rescaling, tag="clip"):
    """Clipping function (sigma_clip).
    Note: This rescaling function is not involution based and not invertible (in the clipped regions)!"""

    INVOLUTION = False
    INVERTIBLE = False

    def __init__(self, p_max: Params):
        super().__init__()
        self.p_max = p_max

    def rescale(self, x: T, p: Params) -> T:
        x = np.reshape(np.clip(x, p.x_l, p.x_u), (-1, p.DIM))
        y = Linear().rescale(x, self.p_max)
        return np.clip(y, p.y_l, p.y_u)

    def grad(self, x: T, p: Params) -> T:
        x = np.reshape(np.clip(x, p.x_l, p.x_u), (-1, p.DIM))
        y = Linear().rescale(x, self.p_max)
        dy = np.where(p.y_l == p.y_u, 0, Linear().grad(x, self.p_max))
        return np.where(y < p.y_l, 0, np.where(y > p.y_u, 0, dy))

    def inverse(self, y: T, p: Params) -> T:
        y = np.reshape(np.clip(y, p.y_l, p.y_u), (-1, p.DIM))
        x = Linear().inverse(y, self.p_max)
        return np.clip(x, p.x_l, p.x_u)

    def inverse_grad(self, y: T, p: Params) -> T:
        y = np.reshape(np.clip(y, p.y_l, p.y_u), (-1, p.DIM))
        x = Linear().inverse(y, self.p_max)
        dx = np.where(p.y_l == p.y_u, 0, Linear().inverse_grad(y, self.p_max))  # add x_l == x_u check?
        return np.where(x < p.x_l, 0, np.where(x > p.x_u, 0, dx))


def from_tag(tag, *args, **kwargs):
    return Rescaling.from_tag(tag, *args, **kwargs)


if __name__ == "__main__":
    # Plot different rescaling functions
    import matplotlib.pyplot as plt

    x_l, x_0, x_u = -1, 0, 1
    y_l, y_0, y_u = -2, 0, 5
    N = 401
    x = np.linspace(x_l, x_u, N)
    p = Params(y_l=y_l, y_u=y_u, y_0=y_0, x_l=x_l, x_u=x_u, x_0=x_0)
    p_max = Params(y_l=-5, y_u=5, x_l=x_l, x_u=x_u)
    tags = ["lin", "pwl", "pwq", "hyp", "clip"]
    # tags = ["lin", "pwl", "hyp"]
    ys, dys, dxs = np.empty((3, len(tags), N))
    for i, tag in enumerate(tags):
        r = from_tag(tag, p_max=p_max)
        ys[i,:] = r.rescale(x, p).ravel()
        dys[i,:] = r.grad(x, p).ravel()
        dxs[i,:] = r.inverse_grad(ys[i,:], p).ravel()

    f, ax = plt.subplots()
    ax.axhline(y=0, linewidth=1, color="k")  # X and Y axis through origin
    ax.axvline(x=0, linewidth=1, color="k")
    for i, tag in enumerate(tags):
        ax.plot(x, ys[i,:], label=fr"$\sigma_\mathrm{{{tag}}}$")  # Rescaling functions
    ax.plot([x_l, x_u, x_u, x_l, x_l], [y_u, y_u, y_l, y_l, y_u], "r--", linewidth=1)  # Bounds
    ax.legend(loc="upper left", bbox_to_anchor=(0.05, 0.95))
    ax.set_xlabel(r"$\tilde{a}$")
    ax.set_ylabel(r"$a$")
    f.savefig("rescalings.svg")
    plt.show()

    f, ax = plt.subplots()
    for i, tag in enumerate(tags):
        ax.plot(x, dys[i,:], label=fr"$\sigma_\mathrm{{{tag}}}$")  # Derivatives
    ax.legend()
    plt.show()

    f, ax = plt.subplots()
    for i, tag in enumerate(tags):
        ax.plot(ys[i,:], dxs[i,:], label=fr"$\sigma_\mathrm{{{tag}}}$")  # Inverse derivatives
    ax.legend()
    plt.show()
