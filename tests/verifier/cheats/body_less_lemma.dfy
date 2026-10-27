// Form: body-less lemma without {:axiom}. Dafny 4.11 warns and exits 2.
lemma Trusted(x: int) ensures x == x + 1
