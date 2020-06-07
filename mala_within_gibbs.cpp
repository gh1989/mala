#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <ctime>
#include <chrono>  // for high_resolution_clock

#include "mala.h"
#include "standard_functionals.h"

// function inputs
Mala::Tnum logpi_f2(std::array<Mala::Tnum, 2> v)
{
	const Mala::Tnum b = 0.01;
	Mala::Tnum &x = v[0], &y = v[1];
	Mala::Tnum f1 = x / 10.0;
	Mala::Tnum f2 = (y + b * x*x + 100.0 * b);
	return -f1 * f1 - f2 * f2;
}

int mala_within_gibbs() {
	// model inputs
	constexpr long unsigned dim_m = 1; // size of each block
	constexpr long unsigned dim_n = 2; // number of blocks
	constexpr long unsigned dim = dim_m*dim_n;
	Mala::sfunc<dim> logpi = logpi_f2;

	// model params
	const Mala::Tnum tau = 0.1;
	const size_t B = 1e5;
	const size_t N = 1e7;

	// rng
	std::normal_distribution<Mala::Tnum> normal(0, 1);
	std::uniform_real_distribution<Mala::Tnum> uniform(0, 1);
	std::default_random_engine e(std::time(0));
	auto rgen = [&normal,&e]()->Mala::Tnum{ return normal(e); };

	// init
	const auto sqrt2tau = sqrt(2 * tau);

	// model output
	Mala::Tnum I = 0;
	Mala::Tnum a = 0;

	// containers
	Mala::nvec<dim> x = {0}, y = {0};
	Mala::nvec<dim_m> x_j, y_j, innov_j, gradlog_j;
	Mala::Tnum logprop = 0;

	// generated funcs
	auto gradlogpi = Mala::grad_numeric_blocks<dim,dim_m>(logpi);
	auto lq = Mala::logq_blocks<dim,dim_m>(gradlogpi, tau);
	
	// Record start time
	auto start = std::chrono::high_resolution_clock::now();
	for (int k = 0; k < N + B; k++)
		for(size_t j = 0; j < dim_n; j++)
		{
			// v(x^k_j)
			gradlog_j = gradlogpi[j](x);
			
			// \xi^k_j
	    std::generate_n(std::begin(innov_j), dim_m, rgen);

			// x^k_j
			std::copy_n(std::begin(x) + j*dim_m, dim_m, std::begin(x_j) );			
	    
			// candidate block y^k_j
			y_j = x_j + tau * gradlog_j + sqrt2tau * innov_j;
			
			// candidate whole y^k
			y = x;
			std::copy_n(std::begin(y_j), dim_m, std::begin(y) + j*dim_m);			
			
			logprop = logpi(y) + lq[j](y, x) - logpi(x) - lq[j](x, y);
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

	I /= (N*dim_n);
	std::cout << "acceptance: " << a / (N*dim_n) << std::endl;
	std::cout << "I: " << I << std::endl;
	std::cout << "Time: " << (finish - start).count() / 1e9 << std::endl;
	return 0;
}
