#include <cmath>
#include <iostream>
#include <random>
#include <ctime>
#include <array>
#include <functional>
#include <algorithm>
#include <iterator>
#include <chrono>  // for high_resolution_clock

#include "mala.h"

// typedefs
typedef double Tnum;
template<long unsigned N> using nvec = std::array<Tnum, N>;
template<long unsigned N> using sfunc = std::function<Tnum(const nvec<N>&)>;
template<long unsigned N> using sfunc_binary = std::function<Tnum(const nvec<N>&, const nvec<N>&)>;
template<long unsigned N> using vfunc = std::function<nvec<N>(const nvec<N>&)>;

// constants
const Tnum infinity = std::numeric_limits<Tnum>::infinity();

// standard functionals
template<long unsigned N>
vfunc<N> grad_numeric(sfunc<N> f, Tnum epsilon = 1e-4)
{
	vfunc<N> Output = [f, epsilon](const nvec<N> &v)-> nvec<N> {
		nvec<N> grdlog;
		const auto base = f(v);
		nvec<N> tau = v;
		for (auto i = 0; i < N; i++)
		{
			tau[i] += epsilon;
			grdlog[i] = (f(tau) - base) / epsilon;
			tau[i] -= epsilon;
		};
		return grdlog;
	};
	return Output;
}

template<long unsigned N>
sfunc_binary<N> logq(vfunc<N> gradlf, Tnum tau)
{
	sfunc_binary<N> output = [gradlf, tau](const nvec<N>& x, const nvec<N>& y)->Tnum {
		Tnum s2 = 4 * tau;
		Tnum net = 0;
		nvec<N> grad = gradlf(x);
		for (auto i = 0; i < N; i++)
		{
			auto t = y[i] - x[i] - tau * grad[i];
			net += t * t;
		}
		return -net / (4.0*tau);
	};
	return output;
};

// function inputs
Tnum logpi_f1(std::array<Tnum, 2> v)
{
	const Tnum b = 0.01;
	Tnum &x = v[0], &y = v[1];
	Tnum f1 = x / 10.0;
	Tnum f2 = (y + b * x*x + 100.0 * b);
	return -f1 * f1 - f2 * f2;
}

int mala_more() {
	const Tnum tau = 0.032;
	constexpr long unsigned dim = 2;

	// rng
	std::normal_distribution<Tnum> normal(0, 1);
	std::uniform_real_distribution<Tnum> uniform(0, 1);
	std::default_random_engine e(time(0));

	sfunc<dim> logpi = logpi_f1;

	const auto sqrt2tau = sqrt(2 * tau);
	nvec<dim> x{ 0,0 }, y{ 0,0 };
	bool burnin = true;
	const size_t B = 1e5;
	const size_t N = 1e7;

	Tnum I = 0;
	Tnum a = 0;

	nvec<dim> gradlog_k, logq_k_num, logq_k_den;
	Tnum logprop = 0;

	auto gradlogpi = grad_numeric(logpi);
	auto lq = logq(gradlogpi, tau);

	// Record start time
	auto start = std::chrono::high_resolution_clock::now();
	for (int k = 0; k < N + B; k++)
	{
		gradlog_k = gradlogpi(x);

		// candidate
		for (auto i = 0; i < dim; i++)
			y[i] = x[i] + tau * gradlog_k[i] + sqrt2tau * normal(e);

		logprop = logpi(y) + lq(y, x) - logpi(x) - lq(x, y);

		if (log(uniform(e)) < logprop)
		{
			x = y;
			if (k > B)	a++;
		}

		if (k > B)
			I += x[0] * x[0] + x[1] * x[1];
	}
	// Record end time
	auto finish = std::chrono::high_resolution_clock::now();


	I /= N;
	std::cout << "acceptance: " << a / N << std::endl;
	std::cout << "I: " << I << std::endl;
	std::cout << "Time: " << (finish - start).count() / 1e9 << std::endl;
	return 0;
}
