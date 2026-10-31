// Golden output for the fixed `find_pattern` in the z-function demo
// (`strings/z_function.py`), assembled with a hand-written spec. The
// `GoldenZFunction` stub stands in for the separately proven callee so that
// `dafny resolve` passes on this file as it stands.
include "../../../src/pcc/pylib/PyLib.dfy"

module GoldenZFunction {
  import opened PyLib

  method z_function(s: string) returns (result: seq<int>)
  {
    return [];
  }
}

module GoldenFindPattern {
  import opened PyLib
  import GoldenZFunction

  predicate pre(pattern: string, text: string)
  {
    true
  }

  predicate spec(pattern: string, text: string, result: int)
    requires pre(pattern, text)
  {
    result >= 0
  }

  method find_pattern(pattern: string, text: string) returns (result: int)
    requires pre(pattern, text)
    ensures spec(pattern, text, result)
  {
    var combined := (pattern + "$") + text;
    var z_result := GoldenZFunction.z_function(combined);
    var n := py_len(pattern);
    return py_count(py_slice(z_result, n, py_len(z_result)), value => value >= n);
  }
}
