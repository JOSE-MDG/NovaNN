# DYNASTRAT: how NovaNN splits threads across its 3 pools

> Design note for the thread-budget split behind `stratify_threads()`
> (`manager.c`): what problem it solves, how the policy works, and which
> knobs to turn when it misbehaves.

## 1. What it solves

NovaNN runs three internal thread pools, and they are not equals. Each one
parallelizes a different kind of work, so each one deserves a different
share of threads — but none may ever sit at zero, and they must not fight
each other:

- **compute (math)** does the heavy number crunching. Highest priority: it
  gets the largest share whenever there is anything worth sharing.
- **autograd (dag)** runs DAG nodes in parallel, but only the independent
  ones. Lowest priority: a couple of threads go a long way here, and any
  extra just adds contention.
- **dtloader (load)** fetches and prepares data. Middle priority: it needs
  comfortably more than the minimum, without hogging the machine.

`T` is the total we have to hand out: the process's logical thread count,
or the global budget the user configured. Any integer `>= 2` works, power
of two or not (`2, 4, 8, 16, 32, 64…` as well as `5, 7, 12, 24, 100…`).
A feel for the expected behavior:

| T | Output (compute, autograd, dtloader) | Reading |
|---|---|---|
| 2 | (1, 1, 1) | intentional overcommit (see §3) |
| 3 | (1, 1, 1) | minimum base |
| 6 | (2, 2, 2) | fairness: too small a total to stratify |
| 8 | (4, 2, 2) | compute pulls ahead, autograd held back |
| 16 | (9, 3, 4) | full stratification |

## 2. Inputs, outputs and parameters

The policy takes a total and a handful of knobs:

- `T`: total threads. Minimum accepted: `2`.
- `m`: floor per pool. Fixed to `MIN_THREADS_PER_GROUP` (`1`).
- `w_target = (w1, w2, w3)`: the target shares for
  (compute, autograd, dtloader): `(0.60, 0.15, 0.25)`. They must add up to
  `1.0` with `w1 > w3 > w2`.
- `EQUAL_TH = 6`: at or below this, split evenly.
- `FULL_TH = 16`: at or above this, split by the target weights.
- `LIGHT_TH = 7`: at or above this, apply the light ordering touch.
- `STRICT_TH = 12`: at or above this, enforce the strict order.

Everything except `T` lives as a constant in `manager.h`
(`DYNASTRAT_EQUAL_TH`, `DYNASTRAT_LIGHT_TH`, `DYNASTRAT_STRICT_TH`,
`DYNASTRAT_FULL_TH`, `DYNASTRAT_W_COMPUTE`, `DYNASTRAT_W_AUTOGRAD`,
`DYNASTRAT_W_DTLOADER`), so retuning never touches `manager.c`.

It returns `(a1, a2, a3)`: the thread counts for
(compute, autograd, dtloader).

## 3. Rules that never break

1. **No starvation:** every pool gets `>= 1`. Always.
2. **Controlled overcommit:** for `T <= 3` the answer is `(1, 1, 1)`, even
   though that sums to 3 when only 2 were asked for. One thread too many
   beats a dead pool.
3. **Conservation:** for `T > 3` the three numbers add up to `T` exactly.
   Rounding never eats a thread.
4. **Anti-collision:** once there are enough threads,
   `a1 > a3 > a2`: compute above dtloader, dtloader above autograd. Small
   totals are allowed their ties (see §6).
5. **Determinism:** same `T`, same parameters, same answer. Every time.

## 4. The core idea in one sentence

Few threads: split evenly. Many threads: split by importance. In between:
blend the weights linearly with `T`, hand out the fractions by largest
remainder, then nudge the result into the right order.

A single fixed weight set can never do both `6 → 2,2,2` and `8 → 4,2,2`
— that pair is what forces the interpolation.

## 5. The algorithm (normative)

```
function DYNASTRAT(T, m=1, w_target=(0.60,0.15,0.25),
                   EQUAL_TH=6, FULL_TH=16):
    # Step 1: base overcommit
    if T <= 3:
        return (1, 1, 1)

    # Step 2: reserve and remainder
    R = T - 3*m
    if R == 0:
        return (m, m, m)

    # Step 3: dynamic weights
    if T <= EQUAL_TH:      factor = 0
    elif T >= FULL_TH:     factor = 1
    else:                  factor = (T - EQUAL_TH) / (FULL_TH - EQUAL_TH)

    w1 = (1/3)*(1-factor) + w_target[1]*factor
    w2 = (1/3)*(1-factor) + w_target[2]*factor
    w3 = (1/3)*(1-factor) + w_target[3]*factor

    # Step 4: proportional split with largest remainders (Hamilton)
    ideal  = (R*w1, R*w2, R*w3)
    base   = (floor(ideal[1]), floor(ideal[2]), floor(ideal[3]))
    rest   = (ideal[1]-base[1], ideal[2]-base[2], ideal[3]-base[3])
    deficit = R - (base[1]+base[2]+base[3])   # is 0, 1 or 2
    # hand 1 extra unit to each of the `deficit` pools with largest `rest`
    # (break exact ties by priority compute > dtloader > autograd)
    extra = indices of the `deficit` largest rests
    for i in extra: base[i] += 1

    (a1, a2, a3) = (base[1]+m, base[2]+m, base[3]+m)

    # Step 5: anti-collision order fixup
    if T >= 12:
        # strict order: repeat until a1 > a3 > a2 (max 10 passes)
        repeat up to 10 times:
            if a1 <= a3 and a3 > m: a1+=1; a3-=1
            if a3 <= a2 and a2 > m: a3+=1; a2-=1
            if a1 > a3 > a2: break
    elif T >= 7:
        # light touch, single pass:
        if a1 <= a3 and a3 > m: a1+=1; a3-=1
        if a3 < a2 and a2 > m:  a3+=1; a2-=1
        if a1 == a3 and a3 > m: a1+=1; a3-=1
        if a3 < a2 and a2 > m:  a3+=1; a2-=1
    else:
        # T <= 6: no fixup, ties stand
        pass

    return (a1, a2, a3)
```

Notes for the `manager.c` implementation:

- Only `ideal` and `rest` need floating point; the rest is integer math.
- `floor` here just truncates downward (nothing is ever negative), so the
  C code casts to unsigned instead and skips `libm` entirely.
- At most 2 extra units change hands, so plain comparisons do the job —
  no sorting needed.
- The fixup never takes a pool below `m`: a donor only gives if it stays
  `>= m`.
- `R < 0` can only happen with `m > 1`. With today's unit minimum it is
  unreachable; the code falls back to an even split rather than failing.

Runs in `O(1)` time and `O(1)` memory for our 3 pools.

## 6. Why the thresholds are 6, 7, 12 and 16

- **`T <= 6`:** there is nothing to stratify with. Opening gaps here would
  just unbalance tiny budgets, so the order is left alone and you get ties
  like `6 → 2,2,2`.
- **`7 <= T <= 11`:** the light touch. Just enough room to lift compute to
  the top, not enough for the full three-level order yet.
- **`T >= 12`:** two 1-thread gaps finally fit above the unit minimum, so
  `a1 > a3 > a2` becomes mandatory.
- **`EQUAL_TH=6 → factor 0`, `FULL_TH=16 → factor 1`:** below 6 every pool
  weighs `1/3`; above 16 the target weights rule; between them the blend
  moves smoothly, no jumps. 16 is the line because it already covers
  mainstream hardware (8 cores / 16 threads and below): most machines get
  full stratification, and bigger ones simply stay on the target weights.

## 7. Worked traces

**`T = 5 → (2, 1, 2)`.**
`R = 2`, `factor = 0`, fair weights, `ideal ≈ (0.67,0.67,0.67)`. All three
remainders tie, `deficit = 2`, so priority hands the extras to compute and
dtloader: `(1,0,1)`, plus the minimum → `(2,1,2)`. No fixup at this size.
Notice dtloader growing before autograd — exactly what `w3 > w2` asks for.

**`T = 6 → (2, 2, 2)`.**
`R = 3`, fair weights, `ideal = (1,1,1)`, nothing left over
(`deficit = 0`), plus the minimum → `(2,2,2)`. No fixup.

**`T = 8 → (4, 2, 2)`.**
`R = 5`, `factor = 2/10 = 0.2`, `w ≈ (0.387, 0.297, 0.317)`,
`ideal ≈ (1.93, 1.48, 1.58)`. Floors take `(1,1,1)`, remainders
`(0.93, 0.48, 0.58)` give the 2 extras to compute and dtloader → `(2,1,2)`,
plus the minimum → `(3,2,3)`. The light fixup sees `a1 <= a3` and moves
one thread over → `(4,2,2)`.

**`T = 16 → (9, 3, 4)`.**
`R = 13`, `factor = 1`, target weights,
`ideal = (7.8, 1.95, 3.25)`. Floors take `(7,1,3)`, remainders
`(0.8, 0.95, 0.25)` give the 2 extras to autograd (`0.95`) and compute
(`0.8`) → `(8,2,3)`, plus the minimum → `(9,3,4)`. Already strictly
ordered, so the fixup has nothing to do.

**`T = 24 → (14, 4, 6)`.**
`R = 21`, `factor = 1`, target weights,
`ideal = (12.6, 3.15, 5.25)`. Floors take `(12,3,5)`; the single extra
goes to the largest remainder (`0.6`, compute) → `(13,3,5)`, plus the
minimum → `(14,4,6)`. Strictly ordered out of the box.

**`T = 2 → (1, 1, 1)`.**
Step 1, straight away. The sum is `3 ≠ 2` on purpose (overcommit).

## 8. Reference table (`manager.h` parameters)

```
 T  | (compute, autograd, dtloader) | note
 2  | (1, 1, 1)                     | overcommit
 3  | (1, 1, 1)                     | base
 4  | (2, 1, 1)                     | compute moves first
 5  | (2, 1, 2)                     | odd; dtloader grows before autograd
 6  | (2, 2, 2)                     | fairness
 7  | (3, 2, 2)                     | odd
 8  | (4, 2, 2)                     | compute x2
 9  | (4, 2, 3)                     | dtloader gets its first extra
12  | (5, 3, 4)                     | strict order visible
13  | (6, 3, 4)                     | odd
16  | (9, 3, 4)                     | full stratification
24  | (14, 4, 6)                    | past FULL_TH; shares hold at target
32  | (19, 5, 8)                    | power of 2
64  | (38, 10, 16)                  | power of 2
100 | (59, 16, 25)                  | non-power-of-2, no special case
```

The pattern is hard to miss: autograd barely grows, dtloader sits in the
middle, compute soaks up nearly all growth. Shares converge to `60/15/25`.

## 9. How to tune it

| Parameter | Constant | Value | Effect |
|---|---|---|---|
| `m` | `MIN_THREADS_PER_GROUP` | `1` | Per-pool floor. Raising it flattens everything (less `R` left to argue over). |
| `w_target` | `DYNASTRAT_W_COMPUTE`, `DYNASTRAT_W_AUTOGRAD`, `DYNASTRAT_W_DTLOADER` | `(0.60, 0.15, 0.25)` | Larger `w1` widens compute's lead. Must sum to `1.0`. |
| `EQUAL_TH` | `DYNASTRAT_EQUAL_TH` | `6` | Higher keeps small budgets fair for longer. |
| `LIGHT_TH` | `DYNASTRAT_LIGHT_TH` | `7` | Higher delays the light touch. |
| `STRICT_TH` | `DYNASTRAT_STRICT_TH` | `12` | Lower enforces the strict order earlier. |
| `FULL_TH` | `DYNASTRAT_FULL_TH` | `16` | Lower reaches full stratification sooner. |

## 10. Edge cases (minimum checks for this policy)

- `T=2 → (1,1,1)`, `T=3 → (1,1,1)`.
- `T=6 → (2,2,2)`, `T=8 → (4,2,2)`, `T=16 → (9,3,4)`,
  `T=24 → (14,4,6)`.
- For every `T > 3`: the numbers add up exactly and no pool drops below
  `m`.
- For every `T >= 12` (with `m=1`): `a1 > a3 > a2`.
- Odd totals get no special treatment: `5, 7, 13, 31, 33` must add up
  exactly and never error.
- `T < 2`: error (`novaInvalidNumThreads`), no split.

## 11. Why this combination works

Three borrowed ideas, one original twist:

- **Proportional shares** (fair queueing): `ideal = R·w` says what each
  pool *deserves*.
- **Largest remainders** (Hamilton/Hare-Niemeyer): floors plus extras to
  the biggest fractions. This honors the *quota rule* — everybody lands on
  their ideal share rounded up or down — and the sum comes out exact.
- **Total-dependent weights** (the actual contribution here): `w(T)` slides
  from fairness to priority as the budget grows. That single sliding vector
  pulls off the `6→tie / 8→stratified` pair no fixed weight set can.

It generalizes to `N` pools the obvious way: `R = T - N·m`, an `N`-weight
vector summing to `1`, largest remainders over `N` fractions, and a fixup
pushing toward whatever priority order you want while respecting `m`.

## 12. How it maps to the code

- `stratify_threads()` (`manager.c`, contract in `manager.h`) is the
  reference implementation: a pure function, no allocation, constant cost,
  safe to call from anywhere.
- Inside, pools are indexed `0` compute, `1` autograd, `2` dtloader —
  the same order as the `StratifiedThreads` fields in `threads.h`.
- Bad inputs (`T < 2`, `status == nullptr`) return `(0, 0, 0)`; with
  `T < 2` the status also carries `novaInvalidNumThreads`.
- `get_stratified_threads()` (`threads.c`) figures out the total — process
  affinity when the user set no budget, the configured budget otherwise —
  and hands it to `stratify_threads()`. The first success is recorded, and
  later calls reuse the record while the global budget stays put.
- `is_stratification_complete()` and `get_last_stratification_result()`
  (`manager.c`, declared in `manager.h`) expose the cache state and the
  recorded budget. Before the first success the record reads `(0, 0, 0)`.
- `is_valid_stratification_result()` and
  `is_valid_latest_stratification_result()` (`threads.c`, declared in
  `threads.h`) reject budgets with an empty group — that covers both the
  `(0, 0, 0)` failure sentinel and any split that would starve a pool.
- `distribute_stratified_threads()` (`threads.c`, declared in
  `threads.h`) validates a budget, then writes it into the per-group
  counters in (compute, autograd, dtloader) order, stopping at the first
  group that refuses it.
- `print_thread_config()` (`threads.c`, declared in `threads.h`) renders
  the budget and the live counters to stdout, in concise or verbose
  form. It guards on `is_thread_config_initialized()`: without a full
  configuration it shows whatever is known with the rest marked
  `NOT INITIALIZED`.
