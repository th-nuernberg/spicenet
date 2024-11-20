from typing import Callable

__c = 0.381966  # c = (3 - sqrt(5)) / 2


def approximate_local_min(interval_start: float, interval_end: float, eps: float, t: float, function: Callable[[float], float]) -> float:
    """
    This function approximates the local minima of the given function in an interval. For more details look up ISBN 978-1-306-35261-1.

    :param interval_start: The included beginning of the interval.
    :param interval_end: the included end of the interval.
    :param eps: t and eps define a tolerance tol − eps | x | + t, and f is never evaluated at two points closer together than tol.
    :param t: t and eps define a tolerance tol − eps | x | + t, and f is never evaluated at two points closer together than tol.
    :param function: A function defined on the interval. The local minima will be approximated for this function.
    """

    v = w = x = interval_start + __c * (interval_end - interval_start)
    e = 0
    fv = fw = fx = function(x)

    tol = 1.0e-6 * (limL + limH) / 2.0

    tol = eps * abs(x) + t
    m = 0.5 * (interval_start + interval_end)
    t2 = 2 * tol

    while True:
        if abs(x - m) > t2 - 0.5 * (interval_end - interval_start):
            p = q = r = 0
            if abs(e) > tol:
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
                if u - interval_start < t2 or interval_end - u < t2:
                    d = tol if x < m else -tol
            else:
                e = (interval_end if x < m else interval_start) - x
                d = __c * e
            u = x + (d if abs(d) >= tol else (tol if d > 0 else -tol))
            fu = function(u)
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
            break
    return fx
