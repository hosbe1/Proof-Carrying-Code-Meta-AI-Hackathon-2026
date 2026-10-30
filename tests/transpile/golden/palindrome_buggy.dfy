// Golden output for the buggy `is_palindrome_recursive` (`if len(s) <= 2: return True` ...),
// assembled with a hand-written spec. `dafny resolve` must pass on this file as it stands.
include "../../../src/pcc/pylib/PyLib.dfy"

module GoldenPalindromeBuggy {
  import opened PyLib

  predicate pre(s: string)
  {
    true
  }

  predicate spec(s: string, result: bool)
    requires pre(s)
  {
    result == (s == py_reversed(s))
  }

  method is_palindrome_recursive(s: string) returns (result: bool)
    requires pre(s)
    ensures spec(s, result)
  {
    if py_len(s) <= 2 {
      return true;
    }
    if py_index(s, 0) != py_index(s, -1) {
      return false;
    }
    var pcc_t1 := is_palindrome_recursive(py_slice(s, 1, -1));
    return pcc_t1;
  }
}
