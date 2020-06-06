#ifndef STANDARD_FUNCTIONALS_H
#define STANDARD_FUNCTIONALS_H

#include <cmath>
#include <array>
#include <functional>

// typedefs
typedef double Tnum;
template<long unsigned N> using nvec = std::array<Tnum, N>;
template<long unsigned N> using sfunc = std::function<Tnum(const nvec<N>&)>;
template<long unsigned N> using sfunc_binary = std::function<Tnum(const nvec<N>&, const nvec<N>&)>;
template<long unsigned N> using vfunc = std::function<nvec<N>(const nvec<N>&)>;

template<long unsigned N> 
nvec<N> operator+(const nvec<N>& a, const nvec<N>& b)
{
	nvec<N> c;
	for(auto i=0;i<N;i++)
		c[i] = a[i]+b[i];
	return c;
}

template<long unsigned N> 
nvec<N> operator-(const nvec<N>& a, const nvec<N>& b)
{
	nvec<N> c;
	for(auto i=0;i<N;i++)
		c[i] = a[i]-b[i];
	return c;
}

template<long unsigned N> 
nvec<N> operator*(const nvec<N>& a, const nvec<N>& b)
{
	nvec<N> c;
	for(auto i=0;i<N;i++)
		c[i] = a[i]*b[i];
	return c;
}

template<long unsigned N> 
nvec<N> operator*(const Tnum& a, const nvec<N>& b)
{
	nvec<N> c;
	for(auto i=0;i<N;i++)
		c[i] = a*b[i];
	return c;
}

template<long unsigned N> 
Tnum inner_product(const nvec<N>& a, const nvec<N>& b)
{
	Tnum c=0;
	for(auto i=0;i<N;i++)
		c += a[i]*b[i];
	return c;
}

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
sfunc_binary<N> logq(vfunc<N> gradlogf, Tnum tau)
{
	sfunc_binary<N> output = [gradlogf, tau](const nvec<N>& x, const nvec<N>& y)->Tnum {
		Tnum s2 = 4 * tau;
		Tnum net = 0;
		nvec<N> grad = gradlogf(x);
		for (auto i = 0; i < N; i++)
		{
			auto t = y[i] - x[i] - tau * grad[i];
			net += t * t;
		}
		return -net / (4.0*tau);
	};
	return output;
};

#endif
