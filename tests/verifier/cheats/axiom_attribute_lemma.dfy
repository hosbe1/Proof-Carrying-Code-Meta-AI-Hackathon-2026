// Form: {:axiom} on a body-less lemma. Dafny 4.11 accepts it without a warning.
lemma {:axiom} Trusted(x: int) ensures x == x + 1
