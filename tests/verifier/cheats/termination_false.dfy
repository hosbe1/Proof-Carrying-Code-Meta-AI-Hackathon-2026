// Form: {:termination false}. Dafny 4.11 ignores it and still checks termination,
// so this terminating function verifies; the scan flags the attribute anyway.
function {:termination false} Cheat(n: int): int {
  n
}
