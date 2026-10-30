// Golden output for `def same(s: str) -> bool: return s[0] == s[-1]`, assembled with a
// hand-written spec. `dafny verify` must pass on this file as it stands.
include "../../../src/pcc/pylib/PyLib.dfy"

module GoldenSameEnds {
  import opened PyLib

  predicate pre(s: string)
  {
    |s| > 0
  }

  predicate spec(s: string, result: bool)
    requires pre(s)
  {
    result == (py_index(s, 0) == py_index(s, -1))
  }

  method same(s: string) returns (result: bool)
    requires pre(s)
    ensures spec(s, result)
  {
    return py_index(s, 0) == py_index(s, -1);
  }
}
