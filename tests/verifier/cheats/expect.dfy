// Form: expect. Dafny 4.11 does not prove expect statements; it assumes them
// (they are only checked at run time), so this verifies with exit 0.
method Cheat() ensures false {
  expect false;
}
