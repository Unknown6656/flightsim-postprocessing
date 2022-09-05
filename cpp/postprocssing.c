#include <math.h>


__declspec(dllexport) float _cdecl inverse_dynamic_range(float const value, float const cap, float const amount)
{
    double const x = value - .5;
    double const x2 = x * x;
    double const x3 = x2 * x;
    double const c = .8 * exp(-10. * x2);
    double const d = 1.2 * (c * 8. * x3 + (1. - c) * (x - x3)) * cap;
    double const result = (d + .5) * amount + (x + .5) * (1. - amount);

    return (float)result;
}
