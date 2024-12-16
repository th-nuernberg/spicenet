from typing import Callable

__c = 0.381966  # c = (3 - sqrt(5)) / 2
__eps = 1.1e-16


def approximate_local_min(interval_start: float,
                          interval_end: float,
                          function: Callable[[float], float],
                          tolerance: float = None,
                          eps: float = None,
                          t: float = None) -> float:
    """
    This function approximates the local minima of the given function in an interval. For more details look up ISBN 978-1-306-35261-1.

    :param tolerance: The tolerance for the approximation.
    :param interval_start: The included beginning of the interval.
    :param interval_end: the included end of the interval.
    :param eps: t and eps define a tolerance tol − eps | x | + t, and f is never evaluated at two points closer together than tol.
    :param t: t and eps define a tolerance tol − eps | x | + t, and f is never evaluated at two points closer together than tol.
    :param function: A function defined on the interval. The local minima will be approximated for this function.
    """
    if not (tolerance is not None or (eps is not None and t is not None)):
        raise TypeError("Either tolerance or (eps and t) must be defined.")

    print(f'start: {interval_start} end: {interval_end}')
    v = w = x = interval_start + __c * (interval_end - interval_start)
    e = 0
    fv = fw = fx = function(x)
    # print(fv)

    # tol = 1.0e-6 * (limL + limH) / 2.0
    tol_radius = tolerance if tolerance is not None else eps * abs(x) + t
    tol_both_directions = 2 * tol_radius

    for i in range(0, 50):
        m = 0.5 * (interval_start + interval_end)
        if tolerance is None:
            tol_radius = eps * abs(x)
            tol_both_directions = 2 * tol_radius
        if abs(x - m) > tol_both_directions - 0.5 * (interval_end - interval_start):
            p = q = r = 0
            if abs(e) > tol_radius:
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2 * (q - r)
                if q > 0:
                    p = -p
                else:
                    q = -q
                r = e
                e = d
            if abs(p) < abs(0.5 * q * r) and p < q * (interval_start - x) and p < q * (interval_end - x):
                d = p / q
                u = x + d
                if u - interval_start < tol_both_directions or interval_end - u < tol_both_directions:
                    d = tol_radius if x < m else -tol_radius
            else:
                e = (interval_end if x < m else interval_start) - x
                d = __c * e
            u = x + (d if abs(d) >= tol_radius else (tol_radius if d > 0 else -tol_radius))
            fu = function(u)
            # print(f'x: {u} result :{fu}')
            if fu <= fx:
                if u < x:
                    interval_end = x
                else:
                    interval_start = x
                v = w
                fv = fw
                w = x
                fw = fx
                x = u
                fx = fu
            else:
                if u < x:
                    interval_start = u
                else:
                    interval_end = u
                if fu <= fw or w == x:
                    v = w
                    fv = fw
                    w = u
                    fw = fu
                elif fu <= fv or v == x or v == w:
                    v = u
                    fv = fu
        else:
            return fx
    # raise Exception("max iterations reached")
    return fx


def approximate_local_min_axenie(interval_start: float,
                                 interval_end: float,
                                 function: Callable[[float], float],
                                 tolerance: float = None) -> float:
    """
    This function approximates the local minima of the given function in an interval. For more details look up ISBN 978-1-306-35261-1.

    :param tolerance: The tolerance for the approximation.
    :param interval_start: The included beginning of the interval.
    :param interval_end: the included end of the interval.
    :param function: A function defined on the interval. The local minima will be approximated for this function.
    """
    assert tolerance > 0.0


    # print(f'interval_start: {interval_start} interval_end: {interval_end} tolerance: {tolerance}')
    current_start = interval_start
    current_end = interval_end
    c = interval_end

    f_start = function(current_start)
    f_end = function(current_end)

    f_c = f_end

    for i in range(0, 3000):
        if (f_end > 0.0 and f_c > 0.0) or (f_end < 0.0 and f_c < 0.0):
            c = current_start
            f_c = f_start
            e = d = current_end - current_start
        if abs(f_c) < abs(f_end):
            current_start = current_end
            current_end = c
            c = current_start
            f_start = f_end
            f_end = f_c
            f_c = f_start
        tol = 2.0 * __eps * abs(current_end) + 0.5 * tolerance
        xm = 0.5 * (c - current_end)
        if abs(xm) <= tol or f_end == 0.0:
            return current_end
        if abs(e) >= tol and abs(f_start) > abs(f_end):
            s = f_end / f_start
            if current_start == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = f_start / f_c
                r = f_end / f_c
                p = s * (2.0 * xm * q * (q - r) - (current_end - current_start) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -1 * q
            p = abs(p)
            min_1 = 3.0 * xm * q - abs(tol * q)
            min_2 = abs(e * q)
            selected_min = min_1 if min_1 < min_2 else min_2
            if 2.0 * p < selected_min:
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        current_start = current_end
        f_start = f_end
        if abs(d) > tol:
            current_end = current_end + d
        else:
            current_end = current_end + (abs(tol) if xm >= 0 else abs(tol) * -1)
        f_end = function(current_end)
    raise Exception("This should not happen. Check your values and think about increasing the allowed iterations.")