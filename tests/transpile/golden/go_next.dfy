// Golden output for `go_next` in the z-function demo (`strings/z_function.py`),
// assembled with a hand-written spec. `dafny resolve` must pass on this file as it stands.
include "../../../src/pcc/pylib/PyLib.dfy"

module GoldenGoNext {
  import opened PyLib

  predicate pre(s: string, i: int, z: seq<int>)
  {
    0 <= i < |s| && |z| == |s|
  }

  predicate spec(s: string, i: int, z: seq<int>, result: bool)
    requires pre(s, i, z)
  {
    result == ((i + py_index(z, i) < py_len(s)) && (py_index(s, py_index(z, i)) == py_index(s, i + py_index(z, i))))
  }

  method go_next(s: string, i: int, z: seq<int>) returns (result: bool)
    requires pre(s, i, z)
    ensures spec(s, i, z, result)
  {
    return (i + py_index(z, i) < py_len(s)) && (py_index(s, py_index(z, i)) == py_index(s, i + py_index(z, i)));
  }
}
