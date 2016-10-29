#!/usr/bin/env python

import newton
import unittest
import numpy as N
import functions as F


class TestNewton(unittest.TestCase):

    def testLinear(self):
        '''test the solution for the linear equation'''
        f = lambda x : 3.0 * x + 6.0
        solver = newton.Newton(f, tol=1.e-15, maxiter=200)
        x = solver.solve(-1)
        self.assertEqual(x.item(), -2.0)

    def testLinear2(self):
        '''test the solution for the 2D system of linear equations'''
        A = N.matrix("1. 2.; 3. 4.")
        x0 = N.matrix("0.2; 0.4")
        sol = N.matrix("0.; 0.")
        def f(x):
            return A*x
        solver = newton.Newton(f, tol=1.e-15, maxiter=20)
        x = solver.solve(x0)
        N.testing.assert_array_almost_equal(x, sol)

    def testLinear3(self):
        '''test the solution for the 3D system of linear equations'''
        A = N.matrix("1. 2. 0. ; 3. 4. 0; 0. 0. 1. ")
        x0 = N.matrix("1.; 1.; 1.")
        b = N.matrix("-1.; -1.; 0.")
        sol = N.matrix("-1.; 1.; 0.")
        def f(x):
            return A * x + b
        solver = newton.Newton(f, tol=1.e-15, maxiter=20)
        x = solver.solve(x0)
        N.testing.assert_array_equal(x, sol)

    def testStep(self):
        '''tests if Step function performs as it should'''
        f = lambda x : N.sin(x)
        solver = newton.Newton(f)
        step_value= solver.step(0)
        self.assertEqual(step_value, 0)

    def testConvergence(self):
        '''tests if the Expection raised once the maxiter reached'''
        p = F.Polynomial([1, 2, -3])
        solver = newton.Newton(p, tol=1.e-15, maxiter=1, Df= p.Jacobian)
        with self.assertRaises(Exception):
            x = solver.solve(2.0)

    def testLogPolynomialSolution(self):
        '''testing that the solution for the LogPolynomial class is correct for both analytical and approx Jacobian'''
        p = F.LogPolynomial([1, -2, -6])
        # solution with analytical Jacob
        solver1 = newton.Newton(p, tol=1.e-15, maxiter=100, Df= p.Jacobian)
        x1 = solver1.solve(-2.0)
        # solution with approximated Jacob
        solver2 = newton.Newton(p, tol=1.e-15, maxiter=100)
        x2 = solver2.solve(-2.0)

        N.testing.assert_array_almost_equal(p(x1), N.matrix("0."))
        N.testing.assert_array_equal(p(x1), p(x2))

    def testPolynomialSolution(self):
        '''testing that the solution for the Polynomial class is correct for both analytical and approx Jacobian'''
        p = F.Polynomial([1, 2, -3])
        # solution with analytical Jacob
        solver1 = newton.Newton(p, tol=1.e-15, maxiter=100, Df= p.Jacobian)
        x1 = solver1.solve(0.5)
        # solution with approximated Jacob
        solver2 = newton.Newton(p, tol=1.e-15, maxiter=100)
        x2 = solver2.solve(0.5)

        N.testing.assert_array_almost_equal(p(x1), N.matrix("0."))
        N.testing.assert_array_equal(p(x1), p(x2))

    def testPolynomial2DSolution(self):
        '''testing that the solution for the 2D Polynomial class is correct for both analytical and approx Jacobian'''
        p = F.Polynomial2D([[2, 3, 1],[-2, 1, 0]])
        # solution with analytical Jacob
        solver1 = newton.Newton(p, tol=1.e-15, maxiter=100, Df= p.Jacobian)
        x1 = solver1.solve(N.matrix([[-1],[-2]]))
        # solution with approximated Jacob
        solver2 = newton.Newton(p, tol=1.e-15, maxiter=100)
        x2 = solver2.solve(N.matrix([[-1],[-2]]))

        N.testing.assert_array_equal(p(x1), N.matrix("0.; 0."))
        N.testing.assert_array_equal(p(x1), p(x2))

    def testUseAnalyticalJacob(self):
        '''Test that the analytical Jacobian is used when supplied'''
        p1 = F.LogPolynomial([1, -2, -6])
        solver1 = newton.Newton(p1, tol=1.e-15, maxiter=100, Df= p1.Jacobian)

        p2 = F.Polynomial([1, 2, -3])
        solver2 = newton.Newton(p2, tol=1.e-15, maxiter=100, Df= p2.Jacobian)

        p3 = F.Polynomial2D([[2, 3, 1],[-2, 1, 0]])
        solver3 = newton.Newton(p3, tol=1.e-5, maxiter=100, Df= p3.Jacobian)

        self.assertTrue(solver1._Df is not None)
        self.assertTrue(solver2._Df is not None)
        self.assertTrue(solver3._Df is not None)

    def testNoSolutions(self):
        '''test the case of no solutions'''
        p = F.Polynomial([1, 2, 3])
        solver2 = newton.Newton(p, tol=1.e-15, maxiter=100, Df= p.Jacobian)
        with self.assertRaises(Exception):
            x = solver2.solve(0.5)

    def testOutOfRadius(self):
        '''test if the exception raised when the iteration goes out of the predefined radius'''
        p = F.LogPolynomial([1, -2, -6])
        # solution with analytical Jacob
        solver = newton.Newton(p, tol=1.e-15, maxiter=100, r = 0.1)
        with self.assertRaises(Exception):
            x = solver.solve(-2.0)

if __name__ == "__main__":
    unittest.main()
