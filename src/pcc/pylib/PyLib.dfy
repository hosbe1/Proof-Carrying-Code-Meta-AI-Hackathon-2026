// Python semantics in Dafny. Every block must:
//   1. carry an `ensures` spec that states Python's behaviour,
//   2. verify with `dafny verify` (no assume, {:axiom} or {:verify false}),
//   3. be tested against CPython in tests/pylib.
// Python errors become preconditions: a caller must prove the error case cannot happen.
module PyLib {

  // a // b in Python: floor division. Dafny's `/` is Euclidean and differs for negative divisors
  // (Python 7 // -2 == -4, Dafny 7 / -2 == -3). Precondition: Python raises ZeroDivisionError for b == 0.
  function py_floordiv(a: int, b: int): int
    requires b != 0
    ensures b > 0 ==> b * py_floordiv(a, b) <= a < b * py_floordiv(a, b) + b
    ensures b < 0 ==> b * py_floordiv(a, b) >= a > b * py_floordiv(a, b) + b
  {
    // Both branches divide by a positive number, where Dafny's `/` rounds down. The asserts state the Euclidean
    // facts (dividend == divisor * quotient + remainder, 0 <= remainder < divisor) for that divisor, so the solver
    // does not search for them.
    if b > 0 then
      (assert a == b * (a / b) + a % b && 0 <= a % b < b; a / b)
    // a / b == (-a) / (-b) as rationals, and -b > 0, so the quotient rounds down here as well.
    else
      (assert -a == (-b) * ((-a) / (-b)) + (-a) % (-b) && 0 <= (-a) % (-b) < -b; (-a) / (-b))
  }

  // a % b in Python: the result has the sign of b. Precondition: Python raises ZeroDivisionError for b == 0.
  function py_mod(a: int, b: int): int
    requires b != 0
    ensures py_mod(a, b) == a - b * py_floordiv(a, b)
    ensures b > 0 ==> 0 <= py_mod(a, b) < b
    ensures b < 0 ==> b < py_mod(a, b) <= 0
  {
    a - b * py_floordiv(a, b)
  }
}
