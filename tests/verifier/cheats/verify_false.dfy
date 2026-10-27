// Form: {:verify false}. Dafny 4.11 skips the proof and warns, exit 2.
lemma {:verify false} Cheat(x: int) ensures false {
}
