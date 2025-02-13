//
// Created by Fabian on 18.01.2025.
//

#ifndef SPICENET_CPP_OPTIMIZER_H
#define SPICENET_CPP_OPTIMIZER_H

#ifndef SPICENET_OPTIMIZER_MAX_ITERATIONS
#define SPICENET_OPTIMIZER_MAX_ITERATIONS 300
#endif

#define SPICENET_OPTIMIZER_EPS 1.1e-16

#include <limits>

#include "SpicenetLogging.h"

#ifdef SPICENET_LOGGING

#include <Arduino.h>

#endif

template<typename T>
T approximateLocalMin(T intervalStart,
                      T intervalEnd,
                      const std::function<T(T)> &function);

template<typename T>
T approximateLocalMin(const T intervalStart,
                      const T intervalEnd,
                      const std::function<T(T)> &function) {
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "vectorScalarMultiplication: T must be a numeric type!");

    T e, d, tol, xm, s, p, q, r, min1, min2;

    const T tolerance = abs((1.0e-6) * (intervalStart + intervalEnd) / 2.0);
    if (tolerance < 0.0) {
#ifdef SPICENET_LOGGING
        LOG_LN("approximateLocalMin: Tolerance must be positive.");
#endif
        return std::numeric_limits<T>::max();
        // TODO: hier passt was noch nicht, was wenn range komplett im negativen liegt
    }

    T currentStart = intervalStart;
    T currentEnd = intervalEnd;
    T c = intervalEnd;

    T fxStart = function(currentStart);
    T fxEnd = function(currentEnd);

    T fxC = fxEnd;

    for (int i = 0; i < SPICENET_OPTIMIZER_MAX_ITERATIONS; ++i) {
        if (((fxEnd > 0.0) && (fxC > 0.0)) || (fxEnd < 0.0 && fxC < 0.0)) {
            c = currentStart;
            fxC = fxStart;
            e = d = currentEnd - currentStart;
        }
        if (abs(fxC) < abs(fxEnd)) {
            currentStart = currentEnd;
            currentEnd = c;
            c = currentStart;
            fxStart = fxEnd;
            fxEnd = fxC;
            fxC = fxStart;
        }
        tol = 2.0 * SPICENET_OPTIMIZER_EPS * abs(currentEnd) + 0.5 * tolerance;
        xm = 0.5 * (c - currentEnd);
        if ((abs(xm) <= tol) || (fxEnd == 0.0)) {
            return currentEnd;
        }
        if ((abs(e) >= tol) && (abs(fxStart) > abs(fxEnd))) {
            s = fxEnd / fxStart;
            if (currentStart == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = fxStart / fxC;
                r = fxEnd / fxC;
                p = s * (2.0 * xm * q * (q - r) - (currentEnd - currentStart) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) {
                q = -q;
            }
            p = abs(p);
            min1 = 3.0 * xm * q - abs(tol * q);
            min2 = abs(e * q);
            if (2.0 * p < (min1 < min2 ? min1 : min2)) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }

        } else {
            d = xm;
            e = d;
        }
        currentStart = currentEnd;
        fxStart = fxEnd;
        if (abs(d) > tol) {
            currentEnd += d;
        } else {
            currentEnd += (xm >= 0 ? abs(tol) : -abs(tol));
        }
        fxEnd = function(currentEnd);
    }
#ifdef SPICENET_LOGGING
    LOG_LN("approximateLocalMin: This should not happen. Check your values and think about increasing the allowed iterations.");
#endif
    return std::numeric_limits<T>::max();
    // throw std::runtime_error("approximateLocalMin: This should not happen. Check your values and think about increasing the allowed iterations.");
}

#endif //SPICENET_CPP_OPTIMIZER_H
