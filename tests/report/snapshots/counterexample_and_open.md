<!-- pcc-check -->
## Proof-carrying code

1 counterexample, 2 open

### ❌ `src/shop.py::checkout` — counterexample
pre: `len(cart.items) > 0` (accepts 50 % of inputs)
Reason: returned -1 instead of 0
translated by AI, tested, not guaranteed
Intent: Charges the cart.
- charges the sum of the prices [docstring src/shop.py]
- an empty cart charges 0 [test tests/test_shop.py::test_empty]
Call: `checkout([Item(3), Item(-1)])`
Expected: the sum of the prices
Actual: -1
```python
def test_checkout_breaks_spec():
    assert checkout([Item(3), Item(-1)]) == 2
```
Possible fix (a mutant that satisfies the spec):
```diff
--- a/src/shop.py
+++ b/src/shop.py
@@
-    return -1
+    return 0
```
with 1 assumption:
- **stripe.charge** (library_contract): charges the given amount
<details><summary>Dafny spec</summary>

```dafny
predicate spec(items: seq<int>, result: int)
```

</details>

### ⚪ `src/cart.py::Cart.total` — open
Reason: budget exhausted
translated by the transpiler
Intent: Returns the total price of the cart.
- the total includes discounts [docstring README.md]
<details><summary>Dafny spec</summary>

```dafny
predicate spec(items: seq<int>, result: int)
```

</details>

### ⚪ `src/cart.py::Cart.discount` — open
pre: `len(self.items) > 0` (accepts 12 % of inputs)
⚠ pre accepts only 12 % of generated inputs
Reason: no proof found
translated by the transpiler
Intent: Returns the discount for the cart.
- the total includes discounts [docstring README.md]
<details><summary>Dafny spec</summary>

```dafny
predicate spec(items: seq<int>, result: int)
```

</details>

Cost $0.10 · 30 s