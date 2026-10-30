predicate pre(s: string) { true }
predicate spec(s: string, result: bool) requires pre(s) { result == (s == py_reversed(s)) }
