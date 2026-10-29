// Form: resource limit overrides. Each attribute raises the solver limits for its member.
function {:rlimit 1000000} A(n: int): int { n }

function {:resource_limit 1000000} B(n: int): int { n }

function {:timeLimit 1000000} C(n: int): int { n }

function {:timeLimitMultiplier 100} D(n: int): int { n }

@ResourceLimit("1000000")
function E(n: int): int { n }

@TimeLimit(1000000)
function F(n: int): int { n }

@TimeLimitMultiplier(100)
function G(n: int): int { n }
