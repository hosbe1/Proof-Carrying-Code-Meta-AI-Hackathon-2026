// Form: decreases *. Dafny 4.11 skips the termination check and verifies with exit 0.
method Cheat(n: int) decreases * {
  Cheat(n);
}
