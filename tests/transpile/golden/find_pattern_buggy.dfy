// Golden output for the buggy `find_pattern` in the z-function demo
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
    var answer := 0;
    var pcc_t1 := py_slice(z_result, n, py_len(z_result));
    var pcc_t2 := 0;
    while pcc_t2 < py_len(pcc_t1)
    {
      var value := py_index(pcc_t1, pcc_t2);
      if value >= n {
        answer := answer + 1;
      }
      pcc_t2 := pcc_t2 + 1;
    }
    return answer;
  }
}
