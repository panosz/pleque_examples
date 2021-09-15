import numpy as np
from scipy.integrate import simpson, trapezoid, quad
from scipy.interpolate import CubicSpline, interp1d, Akima1DInterpolator


class FourierCalculatorBase():

    @staticmethod
    def close_period_in_samples(x, y):
        """
        Appends elements in x and y so that the returned vectors sample a whole
        2*pi period.
        """
        x = np.array(x)
        y = np.array(y)

        if np.abs(x[0] - x[-1]) < 2 * np.pi:
            x = np.append(x, x[0]+2*np.pi)
            y = np.append(y, y[0])

        return x, y



class SimpsonFourierCalculator(FourierCalculatorBase):
    @staticmethod
    def Cn(x, y, n):
        """
        Returns the Cn coefficient of the fourier series for y
        """
        out = simpson(y * np.cos(n*x), x)

        if n == 0:
            return out / (2 * np.pi)

        return out / np.pi

    @staticmethod
    def Sn(x, y, n):
        """
        Returns the Sn coefficient of the fourier series for y
        """
        return simpson(y * np.sin(n*x), x) / np.pi

    def __call__(self, x, y, nmax):

        x, y = self.close_period_in_samples(x, y)

        c_coefs = np.zeros(nmax+1)
        s_coefs = np.zeros(nmax+1)

        for n in range(nmax+1):
            c_coefs[n] = self.Cn(x, y, n)

        for n in range(1, nmax+1):
            s_coefs[n] = self.Sn(x, y, n)

        return c_coefs, s_coefs

class InterpFourierCalculatorBase(FourierCalculatorBase):
    def interpolate(self,x,y):
        raise NotImplemented

    @staticmethod
    def Cn(f, n):
        """
        Returns the Cn coefficient of the fourier series for y
        """
        out, *_ = quad(lambda x: f(x) * np.cos(n*x), 0, 2*np.pi)

        if n == 0:
            return out / (2 * np.pi)

        return out / np.pi

    @staticmethod
    def Sn(f, n):
        """
        Returns the Sn coefficient of the fourier series for y
        """
        out, *_ = quad(lambda x: f(x) * np.sin(n*x), 0, 2*np.pi)
        return out / np.pi

    def __call__(self, x, y, nmax):

        x, y = self.close_period_in_samples(x, y)
        csi = self.interpolate(x, y)
        c_coefs = np.zeros(nmax+1)
        s_coefs = np.zeros(nmax+1)

        for n in range(nmax+1):
            c_coefs[n] = self.Cn(csi, n)

        for n in range(1, nmax+1):
            s_coefs[n] = self.Sn(csi, n)

        return c_coefs, s_coefs


class QuadFourierCalculator(InterpFourierCalculatorBase):
    def interpolate(self, x, y):
        return CubicSpline(x, y, bc_type="periodic")

class LinearFourierCalculator(InterpFourierCalculatorBase):
    def interpolate(self, x, y):
        return interp1d(x, y)

class AkimaFourierCalculator(InterpFourierCalculatorBase):
    def interpolate(self, x, y):
        return Akima1DInterpolator(x, y)







if __name__ == "__main__":
    def fun(x):
        return 5 * np.cos(3 * x) - np.sin(3*x)

    fc = QuadFourierCalculator()

    x = np.sort(2 * np.pi * np.random.random(1000))

    if not 0 in x:
        x = np.insert(x, 0, 0)

    if not 2*np.pi in x:
        x = np.append(x, 2*np.pi)

    y = fun(x)

    C, S = fc(x, y, 5)

    xg = np.linspace(0,2*np.pi, num=1000, endpoint=False)
    xg = xg+np.pi/2

    yg = fun(xg)

    Cg, Sg = fc(xg, yg, 5)


