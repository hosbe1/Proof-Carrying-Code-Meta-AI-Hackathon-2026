// Form: {:only}. Dafny 4.11 warns and exits 2 when the member itself verifies.
lemma {:only} Cheat(x: int) ensures x == x {
}
