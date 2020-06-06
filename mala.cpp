#include <cmath>
#include <iostream>
#include <random>
#include <chrono>  // for high_resolution_clock

#include "mala.h"

const size_t B = 1e5;
const size_t N = 1e7;
typedef double Tnum;
const Tnum b = 0.01;
const Tnum tau = 0.032;
static const Tnum epsilon = std::numeric_limits<Tnum>::min();
const Tnum M_PI = 3.141592653589793238462643383279;

Tnum uniform()
{
	double u = 0;
	while (u <= epsilon)
	{
		u = rand() * (1.0 / RAND_MAX);
	}
	return u;
}

// logpi also gradlogpi
template <bool _Y = 0, bool _Grad = 0>
Tnum logpi(Tnum x, Tnum y) {
	if (!_Grad)
	{
		Tnum f1 = x / 10.0;
		Tnum f2 = (y + b * x * x + 100.0 * b);
		return -f1 * f1 - f2 * f2;
	}
	if (_Y)
		return -200 * b + 2 * b*x*x + 2 * y;
	return x / 50.0 + 4 * b*x*(b*x*x - 100 * b + y);
}

// The matrix { {c, 0}, {0, 1} } also { {sqrt(c), 0}, {0, 1} }
template<bool _Y, bool _Sqrt = 0>
Tnum A(Tnum x, Tnum y, Tnum c = 9) {
	if (!_Y)
		return (_Sqrt ? sqrt(c) : c) * x;
	return y;
};

// q(x'|x) where x = (x1, y1) and x' = (x2, y2)
Tnum logq(Tnum x1, Tnum y1, Tnum x2, Tnum y2)
{
	Tnum s2 = 4 * tau;
	Tnum mx = (x2 - x1 - tau * logpi<0, true>(x1, y1));
	Tnum my = (y2 - y1 - tau * logpi<1, true>(x1, y1));
	Tnum norm2 = mx * mx + my * my;
	return (-1.0 / s2) * norm2;
};

Tnum boxmuller(Tnum mu = 0, Tnum sigma = 1) {
	Tnum u1 = uniform();
	Tnum u2 = uniform();
	Tnum z = sqrt(-2 * log(u1))*cos(2 * M_PI * u2);
	return sigma * z + mu;
}

int mala() {
	const auto sqrt2tau = sqrt(2 * tau);
	Tnum x1 = 0, y1 = 0, x2, y2;
	Tnum xprime = 0;
	Tnum dX1, dX2;
	Tnum logalpha = 0;
	Tnum logu = 0;
	bool burnin = true;

	Tnum I = 0;
	Tnum a = 0;

	Tnum u1, u2, Wx, Wy, _glpix, _glpiy;
	Tnum logpix2, logqx1x2, logpix1, logqx2x1;
	Tnum sqrtA = 3;

	// Record start time
	auto start = std::chrono::high_resolution_clock::now();
	for (int k = 0; k < N + B; k++)
	{
		logu = log(uniform());
		Wx = boxmuller();
		Wy = boxmuller();
		_glpix = logpi<0, true>(x1, y1);
		_glpiy = logpi<1, true>(x1, y1);
		x2 = x1 + tau * A<0>(_glpix, _glpiy) + sqrt2tau * A<0, true>(Wx, Wy);
		y2 = y1 + tau * A<1>(_glpix, _glpiy) + sqrt2tau * A<1, true>(Wx, Wy);
		logpix2 = logpi(x2, y2);
		logqx2x1 = logq(x2, y2, x1, y1);
		logpix1 = logpi(x1, y1);
		logqx1x2 = logq(x1, y1, x2, y2);
		logalpha = logpix2 + logqx2x1 - logpix1 - logqx1x2;

		if (logu < logalpha)
		{
			x1 = x2;
			y1 = y2;
			if (k > B)	a++;
		}

		if (k > B)
			I += x1 * x1 + y1 * y1;
	}

	// Record end time
	auto finish = std::chrono::high_resolution_clock::now();

	I /= N;
	std::cout << "acceptance: " << a / N << std::endl;
	std::cout << "I: " << I << std::endl;
	std::cout << "Time: " << (finish - start).count() << std::endl;
	return 0;
}

int main()
{
	mala();
	mala_more();
	system("PAUSE");
}