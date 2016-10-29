#!/usr/bin/env python

import functions as F
import numpy as N
import unittest

class TestFunctions(unittest.TestCase):

    def testApproxJacobian1(self):
        '''test if the Jacobian is approximated correctly for the linear equation'''
        slope = 3.0
        def f(x):
            return slope * x + 5.0
        x0 = 2.0
        dx = 1.e-3
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (1,1))
        self.assertAlmostEqual(Df_x.item(), slope)

    def testApproxJacobian2(self):
        '''test if the Jacobian is approximated correctly for the 2D system of linear equations'''
        A = N.matrix("1. 2.; 3. 4.")
        def f(x):
            return A * x
        x0 = N.matrix("5; 6")
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (2,2))
        N.testing.assert_array_almost_equal(Df_x, A)

    def testApproxJacobian3(self):
        '''test if the Jacobian is approximated correctly for the 3D system of linear equations'''
        A = N.matrix([[1, 2, 3], [3, 4, 3], [3, 4, 5]])
        def f(x):
            return A * x
        x0 = (N.matrix([1, 2, 3])).transpose()
        dx = 1.e-6
        Df_x = F.ApproximateJacobian(f, x0, dx)
        self.assertEqual(Df_x.shape, (3,3))
        N.allclose(Df_x, A)

    def testPolynomial(self):
        '''test if the polynomial function computed correctly'''

        p = F.Polynomial([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), x**2 + 2*x + 3)

    def testLogPolynomial(self):
        '''test if the log(polynomial) function computed correctly'''

        p = F.LogPolynomial([1, 2, 3])
        for x in N.linspace(-2,2,11):
            self.assertEqual(p(x), N.log(x**2 + 2*x + 3))

    def testWrongInputLogPolynomial(self):
        '''test exception is raised if the point is out of the function domain'''
        p = F.LogPolynomial([1, -2, -6])
        x = 2.0
        with self.assertRaises(Exception):
            val = p(x)

    def testPolynomial2D(self):
        '''test if the 2D system of polynomials computed correctly'''
        p = F.Polynomial2D([[1, 2, 3],[1, 0, 0]])

        def expected(x):
            f1 = x[0] + 2*x[1] + 3
            f2 = x[0]
            return N.matrix([[f1[0,0]], [f2[0,0]]])

        for (x,y) in zip(N.linspace(-2,2,11), N.linspace(-2,2,11)):
            point = N.matrix([[x],[y]])
            N.testing.assert_array_equal(p(point), expected(point))

    def testAnalyticalJacobPolynomial(self):
        '''test the correctness of the analytical jacobian for the Polynomial class
         against approximated value and against computed value'''

        p = F.Polynomial([1, 2, 3])

        def expected(x):
            return 2*x+2

        for x in N.linspace(-2,2,11):
            Df_x = F.ApproximateJacobian(p, x, 1e-9)
            self.assertTrue(N.linalg.norm(Df_x -p.Jacobian(x)) < 0.1)
            self.assertEqual(p.Jacobian(x), expected(x))

    def testAnalyticalJacobLogPolynomial(self):
        '''test the correctness of the analytical jacobian for the Log Polynomial class
         against approximated value and against computed value'''

        p = F.LogPolynomial([1, -2, -6])

        def expected(x):
            return (2*x-2)/(x*x-2*x-6)

        for x in N.linspace(4,6,10):
            Df_x = F.ApproximateJacobian(p, x, 1e-9)
            self.assertTrue(N.linalg.norm(Df_x -p.Jacobian(x)) < 0.1)
            self.assertAlmostEqual(p.Jacobian(x), expected(x))

    def testAnalyticalJacobPolynomial2D(self):
        '''test the correctness of the analytical jacobian for the 2D Polynomial class
         against approximated value and against computed value'''

        p = F.Polynomial2D([[2, 3, 1],[-2, 1, 0]])

        def expected(x):
            return N.matrix([[2, 3], [-2, 1]])

        for (x,y) in zip(N.linspace(-2,2,11), N.linspace(-2,2,11)):
            point = N.matrix([[x],[y]])
            Df_x  = F.ApproximateJacobian(p, point, 1e-9)
            self.assertTrue(N.linalg.norm(Df_x -p.Jacobian([x,y])) < 0.1)
            N.testing.assert_array_equal(p.Jacobian([x,y]), expected([x,y]))


if __name__ == '__main__':
    unittest.main()


