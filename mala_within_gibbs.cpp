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

int main() {
	// input func
	constexpr long unsigned dim_m = 1; // size of each block
	constexpr long unsigned dim_n = 2; // number of blocks
	constexpr long unsigned dim = dim_m*dim_n;
	sfunc<dim> logpi = logpi_f1;

	// model params
	const Tnum tau = 0.01;
	const size_t B = 1e5;
	const size_t N = 1e6;

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
	nvec<dim> x = {0}, y = {0};
	nvec<dim_m> x_j, y_j, innov_j, gradlog_j;
	Tnum logprop = 0;

	// generated funcs
	auto gradlogpi = grad_numeric_blocks<dim,dim_m>(logpi);
	auto lq = logq_blocks<dim,dim_m>(gradlogpi, tau);
	
	// Record start time
	auto start = std::chrono::high_resolution_clock::now();
	for (int k = 0; k < N + B; k++)
		for(size_t j = 0; j < dim_n; j++)
		{
			// subvector x_j
			std::copy(begin(x_j), end(x_j), begin(x) + j*dim_m);
			
			gradlog_j = gradlogpi[j](x);
	    std::generate_n(begin(innov_j), dim_m, rgen);
	    
			// candidate block
			y_j = x_j + tau * gradlog_j + sqrt2tau * innov_j;
			// candidate whole
			y = x;
			std::copy(begin(y) + j*dim_m, begin(y)+(j+1)*dim_m, begin(y_j));			
			
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
