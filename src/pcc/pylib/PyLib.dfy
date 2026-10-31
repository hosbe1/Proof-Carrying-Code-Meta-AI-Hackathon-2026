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

  // len(xs) in Python: the number of elements; the empty sequence has length 0, so there is no error case.
  function py_len<T>(xs: seq<T>): int
    ensures py_len(xs) == |xs|
  {
    |xs|
  }

  // xs[i] in Python: negative indices count from the end; indices outside [-|xs|, |xs|) raise IndexError.
  function py_index<T>(xs: seq<T>, i: int): T
    requires -|xs| <= i < |xs|
    ensures i >= 0 ==> py_index(xs, i) == xs[i]
    ensures i < 0 ==> py_index(xs, i) == xs[|xs| + i]
  {
    if i >= 0 then xs[i] else xs[|xs| + i]
  }

  // xs[start:stop] in Python with step 1: negative bounds count from the end and are clamped to [0, |xs|].
  function py_slice<T>(xs: seq<T>, start: int, stop: int): seq<T>
    ensures var loRaw := if start < 0 then start + |xs| else start;
      var hiRaw := if stop < 0 then stop + |xs| else stop;
      var lo := if loRaw < 0 then 0 else if loRaw > |xs| then |xs| else loRaw;
      var hi := if hiRaw < 0 then 0 else if hiRaw > |xs| then |xs| else hiRaw;
      |py_slice(xs, start, stop)| == (if hi <= lo then 0 else hi - lo)
    ensures var loRaw := if start < 0 then start + |xs| else start;
      var hiRaw := if stop < 0 then stop + |xs| else stop;
      var lo := if loRaw < 0 then 0 else if loRaw > |xs| then |xs| else loRaw;
      var hi := if hiRaw < 0 then 0 else if hiRaw > |xs| then |xs| else hiRaw;
      forall i :: 0 <= i < |py_slice(xs, start, stop)| ==> py_slice(xs, start, stop)[i] == xs[lo + i]
  {
    var loRaw := if start < 0 then start + |xs| else start;
    var hiRaw := if stop < 0 then stop + |xs| else stop;
    var lo := if loRaw < 0 then 0 else if loRaw > |xs| then |xs| else loRaw;
    var hi := if hiRaw < 0 then 0 else if hiRaw > |xs| then |xs| else hiRaw;
    if hi <= lo then [] else xs[lo..hi]
  }

  // xs[::-1] in Python: the elements in reverse order; the empty sequence reverses to itself.
  function py_reversed<T>(xs: seq<T>): seq<T>
    ensures |py_reversed(xs)| == |xs|
    ensures forall i :: 0 <= i < |xs| ==> py_reversed(xs)[i] == xs[|xs| - 1 - i]
  {
    seq(|xs|, i requires 0 <= i < |xs| => xs[|xs| - 1 - i])
  }

  // min(a, b) in Python for two ints: the smaller of the two (the first on ties).
  function py_min(a: int, b: int): int
    ensures py_min(a, b) == (if a <= b then a else b)
    ensures py_min(a, b) <= a && py_min(a, b) <= b
    ensures py_min(a, b) == a || py_min(a, b) == b
  {
    if a <= b then a else b
  }

  // sum(1 for v in xs if p(v)) in Python: the number of elements satisfying p.
  function py_count<T>(xs: seq<T>, p: T -> bool): int
    ensures 0 <= py_count(xs, p) <= |xs|
    decreases |xs|
  {
    if |xs| == 0 then 0 else (if p(xs[0]) then 1 else 0) + py_count(xs[1..], p)
  }

  // xs[i] = v in Python: negative indices count from the end; indices outside
  // [-|xs|, |xs|) raise IndexError. The result has the same length as xs with
  // only the normalized position changed.
  function py_set_index<T>(xs: seq<T>, i: int, v: T): seq<T>
    requires -|xs| <= i < |xs|
    ensures |py_set_index(xs, i, v)| == |xs|
    ensures forall j :: 0 <= j < |xs| ==> py_set_index(xs, i, v)[j] == (if j == (if i >= 0 then i else |xs| + i) then v else xs[j])
  {
    var k := if i >= 0 then i else |xs| + i;
    xs[k := v]
  }
}
