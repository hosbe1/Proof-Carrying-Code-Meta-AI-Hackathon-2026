// Clean: a lemma with a proof and a method with a loop invariant. The scan reports nothing.
lemma DoubleIsEven(n: int) ensures (2 * n) % 2 == 0 {
  assert 2 * n == n + n;
}

method CountUp(n: int) returns (c: int)
  requires n >= 0
  ensures c == n
{
  c := 0;
  while c < n
    invariant 0 <= c <= n
  {
    c := c + 1;
  }
}
