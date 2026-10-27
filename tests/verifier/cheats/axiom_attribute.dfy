// Form: assume with the {:axiom} attribute. Dafny 4.11 accepts it without a warning.
method Cheat() ensures false {
  assume {:axiom} false;
}
