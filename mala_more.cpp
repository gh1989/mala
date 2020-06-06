#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <ctime>
#include <chrono>  // for high_resolution_clock

#include "standard_functionals.h"

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
	// input func
	constexpr long unsigned dim = 2;
	sfunc<dim> logpi = logpi_f1;

	// model params
	const Tnum tau = 0.2;
	const size_t B = 1e5;
	const size_t N = 1e7;

	// rng
	std::normal_distribution<Tnum> normal(0, 1);
	std::uniform_real_distribution<Tnum> uniform(0, 1);
	std::default_random_engine e(std::time(0));
  auto rgen = [&normal,&e]()->Tnum{ return normal(e); };

	// init
	const auto sqrt2tau = sqrt(2 * tau);

	// model output
	Tnum I = 0;
	Tnum a = 0;

	// containers
	nvec<dim> x{ 0,0 }, y{ 0,0 };
	nvec<dim> gradlog_k, logq_k_num, logq_k_den,innov_k;
	Tnum logprop = 0;

	// generated funcs
	auto gradlogpi = grad_numeric(logpi);
	auto lq = logq(gradlogpi, tau);

	// Record start time
	auto start = std::chrono::high_resolution_clock::now();
	for (int k = 0; k < N + B; k++)
	{
		gradlog_k = gradlogpi(x);
    std::generate_n(begin(innov_k), dim, rgen);

		// candidate
		y = x + tau * gradlog_k + sqrt2tau * innov_k;
		
		logprop = logpi(y) + lq(y, x) - logpi(x) - lq(x, y);
		if (log(uniform(e)) < logprop)
		{
			x = y;
			if (k > B)
				a++;
		}

		if (k > B)
			I += inner_product(x,x);
	}
	// Record end time
	auto finish = std::chrono::high_resolution_clock::now();

	I /= N;
	std::cout << "acceptance: " << a / N << std::endl;
	std::cout << "I: " << I << std::endl;
	std::cout << "Time: " << (finish - start).count() / 1e9 << std::endl;
	return 0;
}
