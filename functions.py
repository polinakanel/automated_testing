import numpy as N

def ApproximateJacobian(f, x, dx=1e-6):
    """Return an approximation of the Jacobian Df(x) as a numpy matrix"""
    try:
        n = len(x)
    except TypeError:
        n = 1
    fx = f(x)
    Df_x = N.matrix(N.zeros((n,n)))
    for i in range(n):

        v = N.matrix(N.zeros((n,1)))
        v[i,0] = dx
        Df_x[:,i] = (f(x + v) - fx)/v[i,0]
    return Df_x

class Polynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = x^2 + 2x + 3,
    and evaluate p(5):

    p = Polynomial([1, 2, 3])
    p(5)"""

    def __init__(self, coeffs):
        self._coeffs = coeffs

    def __repr__(self):
        return "Polynomial(%s)" % (", ".join([str(x) for x in self._coeffs]))

    def f(self,x):
        ans = self._coeffs[0]
        for c in self._coeffs[1:]:
            ans = x*ans + c

        return ans

    def Jacobian(self,x):
        upd_coeff = []
        n = len(self._coeffs)
        for i,c in enumerate(self._coeffs[:-1]):
            upd_coeff.append( c*(n - 1 - i) )
        f2 = Polynomial(upd_coeff)
        return f2(x)

    def __call__(self, x):
        return self.f(x)

class LogPolynomial(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial p(x) = log( x^2 + 2x + 3 ),
    and evaluate p(5):

    p = Polynomial([1, 2, 3])
    p(5)"""

    def __init__(self, coeffs):
        self._coeffs = coeffs
        self._poly = Polynomial(self._coeffs)

    def __repr__(self):
        return "Exponential Polynomial(%s)" % (", ".join([str(x) for x in self._coeffs]))

    def f(self,x):
        if self._poly(x) <= 0:
            raise Exception("Point out of the function domain. Choose another one.")
        return N.log(self._poly(x))

    def df(self,x):
        return self._poly.Jacobian(x)

    def Jacobian(self,x):
        '''Analytical Jacobian'''
        # J = p'(x)*exp(p(x))
        return self.df(x) / self._poly(x)

    def __call__(self, x):
        return self.f(x)

class Polynomial2D(object):
    """Callable polynomial object.

    Example usage: to construct the polynomial
    p1(x) = 2*x1 + 3*x2 +1,
    p2(x) = -2*x1 + x2
    and evaluate p([1,0]):

    p = Polynomial2D([[2, 3, 1],[-2, 1, 0]])
    p([1,0])"""

    def __init__(self, coeffs):
        if len(coeffs) != 2:
            raise Exception("The number of coefficients is incorrect")

        self._coeffs = N.matrix(coeffs)
    def f1(self,x):
        return self._coeffs[0][0]*x[0].item() + self._coeffs[0][1]*x[1].item() + self._coeffs[0][2].item()

    def f2(self,x):
        return self._coeffs[1][0]*x[0] + self._coeffs[1][1]*x[1] + self._coeffs[1][2]

    def Jacobian(self,x):
        '''Analytical Jacobian'''
        # J = p'(x)*exp(p(x))
        return self._coeffs[:,:2]

    def __call__(self, x):
        val = self._coeffs[:,:2]*x + self._coeffs[:,2]
        return val