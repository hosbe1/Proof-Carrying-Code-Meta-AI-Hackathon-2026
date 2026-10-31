// Expected spec-mode transpiler output (#30) for the palindrome spec
// `result == (s == s[::-1])`, kept as a fixture so the spec translation
// check does not wait for the transpiler. `s[::-1]` becomes an explicit
// quantifier over the indices, which the prover can use directly.
predicate pre(s: string)
{
  true
}

predicate spec(s: string, result: bool)
  requires pre(s)
{
  result == (forall i :: 0 <= i < |s| ==> s[i] == s[|s| - 1 - i])
}
