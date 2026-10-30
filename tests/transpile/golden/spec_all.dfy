predicate pre(xs: seq<int>) { true }
predicate spec(xs: seq<int>, result: int) requires pre(xs) { forall i :: 0 <= i < |xs| ==> xs[i] <= result }
