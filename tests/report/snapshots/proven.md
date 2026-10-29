<!-- pcc-check -->
## Proof-carrying code

1 proven

### ✅ `src/cart.py::Cart.total` — proven
pre: `len(self.items) > 0` (accepts 93 % of inputs)
translated by the transpiler
Intent: Returns the total price of the cart.
- the total includes discounts [docstring README.md]
- an empty cart totals 0 [test tests/test_cart.py::test_empty]
proven under 2 assumptions:
- **json.loads** (library_contract): returns the parsed value
- **pricing.discount** (unproven_callee): returns a non-negative discount
<details><summary>Dafny spec</summary>

```dafny
predicate spec(items: seq<int>, result: int)
```

</details>

Cost $0.42 · 3 min 10 s