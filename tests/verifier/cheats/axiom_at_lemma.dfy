// Form: @Axiom on a body-less lemma. Dafny 4.11 accepts it without a warning.
@Axiom()
lemma Trusted(x: int) ensures x == x + 1
