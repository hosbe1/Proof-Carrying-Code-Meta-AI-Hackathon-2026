predicate pre(xs: seq<int>) { py_len(xs) > 0 }
predicate spec(xs: seq<int>, result: int) requires pre(xs) { result == py_index(xs, 0) }
