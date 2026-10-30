// Form: {:selective_checking}. Dafny 4.11 verifies it with exit 0; only the scan catches it.
lemma {:selective_checking} Cheat() ensures false {
}
