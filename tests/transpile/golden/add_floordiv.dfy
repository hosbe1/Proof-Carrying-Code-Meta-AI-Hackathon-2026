// Golden output for `def add(a: int, b: int) -> int: return a // b`, assembled with a
// hand-written spec. `dafny verify` must pass on this file as it stands.
include "../../../src/pcc/pylib/PyLib.dfy"

module GoldenAddFloordiv {
  import opened PyLib

  predicate pre(a: int, b: int)
  {
    b != 0
  }

  predicate spec(a: int, b: int, result: int)
    requires pre(a, b)
  {
    result == py_floordiv(a, b)
  }

  method add(a: int, b: int) returns (result: int)
    requires pre(a, b)
    ensures spec(a, b, result)
  {
    return py_floordiv(a, b);
  }
}
