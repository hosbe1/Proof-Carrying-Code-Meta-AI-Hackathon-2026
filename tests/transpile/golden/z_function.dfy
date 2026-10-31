// Golden output for `z_function` in the z-function demo (`strings/z_function.py`),
// assembled with a hand-written spec. The `GoldenGoNext` stub stands in for the
// separately proven callee so that `dafny resolve` passes on this file as it stands.
include "../../../src/pcc/pylib/PyLib.dfy"

module GoldenGoNext {
  import opened PyLib

  method go_next(s: string, i: int, z: seq<int>) returns (result: bool)
  {
    return false;
  }
}

module GoldenZFunction {
  import opened PyLib
  import GoldenGoNext

  predicate pre(s: string)
  {
    true
  }

  predicate spec(s: string, result: seq<int>)
    requires pre(s)
  {
    |result| == |s|
  }

  method z_function(s: string) returns (result: seq<int>)
    requires pre(s)
    ensures spec(s, result)
  {
    var n := py_len(s);
    var z := seq(n, i => 0);
    var l, r := 0, 0;
    var i := 1;
    var pcc_t1 := n;
    while i < pcc_t1
    {
      if i <= r {
        z := py_set_index(z, i, py_min((r - i) + 1, py_index(z, i - l)));
      }
      while true
      {
        var pcc_c := GoldenGoNext.go_next(s, i, z);
        if !pcc_c { break; }
        z := py_set_index(z, i, py_index(z, i) + 1);
      }
      if (i + py_index(z, i)) - 1 > r {
        l, r := i, (i + py_index(z, i)) - 1;
      }
      i := i + 1;
    }
    return z;
  }
}
