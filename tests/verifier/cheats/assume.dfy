// Form: plain assume. Dafny 4.11 warns and exits 2 without --allow-warnings.
method Cheat() ensures false {
  assume false;
}
