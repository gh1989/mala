#ifndef STANDARD_FUNCTIONALS_H
#define STANDARD_FUNCTIONALS_H

#include <cmath>
#include <array>
#include <functional>

namespace Mala{
	// typedefs
	typedef double Tnum;
	template<long unsigned N> using nvec = std::array<Tnum, N>;
	template<long unsigned N> using sfunc = std::function<Tnum(const nvec<N>&)>;
	template<long unsigned N> using sfunc_binary = std::function<Tnum(const nvec<N>&, const nvec<N>&)>;
	template<long unsigned N, long unsigned M=N> using vfunc = std::function<nvec<M>(const nvec<N>&)>;
	template<long unsigned N, long unsigned M=N> using vfunc_blocks = std::array<vfunc<N,M>,N/M>;
	template<long unsigned N, long unsigned M=N> using sfunc_binary_blocks = std::array<sfunc_binary<N>,N/M>;


	// constants
	const Tnum infinity = std::numeric_limits<Tnum>::infinity();

	// standard functionals
	template<long unsigned N, long unsigned M=N, long unsigned P=0>
	vfunc<N,M> grad_numeric(sfunc<N> f, Tnum epsilon = 1e-4, long unsigned pos=P)
	{
		vfunc<N,M> Output = [f, epsilon,pos](const nvec<N> &v)-> nvec<M> {
			nvec<M> grdlog;
			const auto base = f(v);
			nvec<N> tau = v;
			auto lim = std::min((pos+1)*M,N);
			for (auto i = pos*M; i<lim; i++)
			{
				tau[i] += epsilon;
				grdlog[i-pos*M] = (f(tau) - base) / epsilon;
				tau[i] -= epsilon;
			};
			return grdlog;
		};
		return Output;
	}

	// standard functionals
	template<long unsigned N, long unsigned M=N>
	vfunc_blocks<N,M> grad_numeric_blocks(sfunc<N> f, Tnum epsilon = 1e-4)
	{
		vfunc_blocks<N,M> Output;
		for(auto i=0;i<(N/M);i++)
			Output[i] = grad_numeric<N,M>(f, epsilon,i);
		return Output;
	}

	template<long unsigned N,long unsigned M=N, long unsigned P=0>
	sfunc_binary<N> logq(vfunc<N,M> gradlogf, Tnum tau, long unsigned pos=P )
	{
		sfunc_binary<N> output = [gradlogf, tau, pos](const nvec<N>& x, const nvec<N>& y)->Tnum {
			Tnum s2 = 4 * tau;
			Tnum net = 0;
			nvec<M> grad = gradlogf(x);
			auto lim = std::min((pos+1)*M,N); 
			for (auto i = pos*M; i < lim; i++)
			{
				auto t = y[i] - x[i] - tau * grad[i-pos*M];
				net += t * t;
			}
			return -net / (4.0*tau);
		};
		return output;
	};

	template<long unsigned N, long unsigned M=N>
	sfunc_binary_blocks<N,M> logq_blocks(vfunc_blocks<N,M> gradlogf, Tnum tau )
	{
		sfunc_binary_blocks<N,M> Output;
		for(auto i=0;i<N/M;i++)
			Output[i] = logq(gradlogf[i], tau, i);
		return Output;
	}
}

template<long unsigned N> 
Mala::nvec<N> operator+(const Mala::nvec<N>& a, const Mala::nvec<N>& b)
{
	Mala::nvec<N> c;
	for(auto i=0;i<N;i++)
		c[i] = a[i]+b[i];
	return c;
}

template<long unsigned N> 
Mala::nvec<N> operator-(const Mala::nvec<N>& a, const Mala::nvec<N>& b)
{
	Mala::nvec<N> c;
	for(auto i=0;i<N;i++)
		c[i] = a[i]-b[i];
	return c;
}

template<long unsigned N> 
Mala::nvec<N> operator*(const Mala::nvec<N>& a, const Mala::nvec<N>& b)
{
	Mala::nvec<N> c;
	for(auto i=0;i<N;i++)
		c[i] = a[i]*b[i];
	return c;
}

template<long unsigned N> 
Mala::nvec<N> operator*(const Mala::Tnum& a, const Mala::nvec<N>& b)
{
	Mala::nvec<N> c;
	for(auto i=0;i<N;i++)
		c[i] = a*b[i];
	return c;
}

template<long unsigned N> 
Mala::Tnum inner_product(const Mala::nvec<N>& a, const Mala::nvec<N>& b)
{
	Mala::Tnum c=0;
	for(auto i=0;i<N;i++)
		c += a[i]*b[i];
	return c;
}

#endif
