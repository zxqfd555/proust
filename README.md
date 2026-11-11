# proust

This repo is a deep-dive on building a robust universal range query engine in C++. The next sections contain the details on the design decisions made and efficient implementation.

In brief, the main lines are:
- We use a segment tree as an efficient structure with `O(log N)` time per both modification and query. We suppose that the reader is familiar with this structure and will only describe the optimized and cache-efficient variants.
- In practice, the `O(log N)` time doesn't mean much: two codes with the same asymptotic may differ up to an order of magnitude in terms of both, throughput and the tail latency. Most of the effort is concentrated on obtaining practical results and reliable benchmarking.

**Shortcuts for the impatient reader.** This is a long document; a few entry points:
- If you only want the numbers, jump to the [benchmark results](#benchmark-results) (split into per-operation latency and aggregate throughput).
- If you want to read the implementation, the two engines live in [`include/engine/robust/engine.h`](include/engine/robust/engine.h), and the operations they are parametrised on are in [`include/engine/operations.h`](include/engine/operations.h).
- If you are after the optimization story — disassembly, the ideas that worked and the ones that did not — see [a round of empirical optimization](#a-round-of-empirical-optimization).
- If you care about *how* the numbers are taken, and why there are two measurement modes at all, the [benchmarking methodology](#benchmarking-methodology) covers it.

## Table of Contents

- [The Task](#the-task)
- [Examples](#examples)
- [Interfaces](#interfaces)
- [Optional operation extensions](#optional-operation-extensions)
  - [Undo (HasUndo)](#undo-hasundo)
  - [FromMany (HasFromMany)](#frommany-hasfrommany)
    - [SIMD discussion](#simd-discussion)
  - [ApplyToMany (HasApplyToMany)](#applytomany-hasapplytomany)
  - [MergeBlock (HasMergeBlock)](#mergeblock-hasmergeblock)
  - [FoldInBlock (HasFoldInBlock)](#foldinblock-hasfoldinblock)
  - [Checking that a hook is there](#checking-that-a-hook-is-there)
- [Building a robust engine](#building-a-robust-engine)
  - [Memory management](#memory-management)
  - [Node topology](#node-topology)
  - [SoA versus AoS](#soa-versus-aos)
  - [Packing densely when possible](#packing-densely-when-possible)
  - [Avoiding branching](#avoiding-branching)
  - [Being smarter about smaller intervals](#being-smarter-about-smaller-intervals)
- [Single point modifications](#single-point-modifications)
  - [Bottlenecks. Fat engine](#bottlenecks-fat-engine)
- [Benchmarking methodology](#benchmarking-methodology)
- [Benchmark results](#benchmark-results)
  - [Per-operation latency (barrier-separated)](#per-operation-latency-barrier-separated)
  - [Aggregate throughput (no barriers)](#aggregate-throughput-no-barriers)
- [External reference implementations (SumInt32, point)](#external-reference-implementations-sumint32-point)
- [A round of empirical optimization](#a-round-of-empirical-optimization)
  - [Measuring on a thermally noisy laptop](#measuring-on-a-thermally-noisy-laptop)
  - [Defaults left on the table](#defaults-left-on-the-table)
  - [Reading the disassembly](#reading-the-disassembly)
  - [What did not work](#what-did-not-work)
  - [Before and after](#before-and-after)
  - [Carrying it to floating point and XOR](#carrying-it-to-floating-point-and-xor)
  - [Borrowing the wide tree's trick: a vectorized-update concept](#borrowing-the-wide-trees-trick-a-vectorized-update-concept)
- [A second round: specializing the descent](#a-second-round-specializing-the-descent)
  - [One-sided fringe walks in the range engine](#one-sided-fringe-walks-in-the-range-engine)
  - [A branchless top-K merge](#a-branchless-top-k-merge)
  - [Squeezing the max-subarray operation](#squeezing-the-max-subarray-operation)
  - [Rejected by measurement](#rejected-by-measurement)
  - [Two measurement artifacts worth recording](#two-measurement-artifacts-worth-recording)
  - [Net effect of the round](#net-effect-of-the-round)
- [A third round: a wide range-assignment engine](#a-third-round-a-wide-range-assignment-engine)
  - [Measuring when the machine is in use](#measuring-when-the-machine-is-in-use)
  - [The wide engine](#the-wide-engine)
  - [Hooks for the wide engine](#hooks-for-the-wide-engine)
  - [Which engine by default](#which-engine-by-default)
  - [The same hook in the point engines: a trade-off](#the-same-hook-in-the-point-engines-a-trade-off)
  - [Building from an array](#building-from-an-array)
  - [Rejected in this round](#rejected-in-this-round)
- [Library Checker: Range Set Range Composite](#library-checker-range-set-range-composite)
- [Ensuring code quality](#ensuring-code-quality)
- [Building and Dependencies](#building-and-dependencies)

## The Task

Suppose you have an array, or rather, an in-memory table, indexed from `0` to `N-1` inclusively. The table abstraction can be described as something like this:

```sql
CREATE TABLE Structure (
    index INTEGER PRIMARY KEY,
    value VALUE_TYPE
);

INSERT INTO Structure (index, value)
VALUES ($i, DefaultValue);  -- for all i in [0, N)
```

Your goal is to be able to perform two types of operations.

The first operation is an *assignment*. Assignment receives the interval bounds and a value that needs to be assigned. This way, in array interpretation it would look as follows:
`Structure[i] = UpdateValue` for all `i` where `RangeStart <= i < RangeEnd` holds. In the SQL syntax, it can be represented as follows:

```sql
UPDATE Structure
SET value = UpdateValue
WHERE index >= RangeStart
  AND index < RangeEnd;
```

The second operation is *computation*. The computation receives a range and computes a predefined associative function `Func` over all values that fit into this range. This way, it results into `Func(Structure[RangeStart], Structure[RangeStart + 1], ..., Structure[RangeEnd - 1])`.

Or, in the SQL pseudocode, that would look as follows:

```sql
SELECT Func(value)
FROM Structure
WHERE index >= RangeStart
  AND index < RangeEnd;
```

The high-level goal for the implementation is to be as generic as possible. To achieve that, we will further heavily utilize C++ templates as a zero-cost compile-time abstraction and the only constraint we put on the function `Func` will be the associativity. That means that `Func(a, Func(b, c))` must be equal to `Func(Func(a, b), c)` - as long as the order of the operands is followed, the computation is correct. Please note that commutativity is not needed.

Please note the fundamental decision here: the type of result must not necessarily be the same as the type of value. This may be counter-intuitive and complicate the conventions a little bit, however, it opens numerous possibilities for the engine, that will be discussed in the next section.

### Examples

The abstract description may be not the clearest. Consider several examples.

The first example is the range update and the range sum. That is, the SQL code for the update is unchanged, but the result computation unwraps into:

```sql
SELECT SUM(value)
FROM Structure
WHERE index >= RangeStart
  AND index < RangeEnd;
```

The picture is similar for the maximum/minimum queries. One can rewrite the querying into:

```sql
SELECT MAX(value), MIN(value)
FROM Structure
WHERE index >= RangeStart
  AND index < RangeEnd;
```

Until now, the examples are commutative and the types of the result and the value coincide. It is time to explore the benefits of supporting the non-commutative functions and different result types.

Consider a harder task, such as "Maximum Sum" from Leetcode. The task asks us to compute the maximum consecutive subsegment sum that is possible. The platform clearly expects the `O(N)` solution where `N` is the number of elements. It is however possible to use our engine to solve it in `O(log N)` on an arbitrary subsegment with an `O(N)`preprocessing. The array can also be modified: any kind of assignment can be done in `O(log N)`. That means overall, that for an array of a million elements, you can solve several millions of different instances of this problem per second.

To reduce the problem to the interface described above, suppose that the value type is `integer`, while the type of the result is a composite structure consisting of four fields: the maximum sum over the range of elements, the maximum sum of any prefix of the range, the maximum sum of any suffix of the range, and the sum of all elements within the range.

The function `Func(LeftOperand, RightOperand)` over the newly create type would look as follows:
1. The sum of the elements in the result is the sum of the same value from `LeftOperand` and `RightOperand`.
2. The maximum prefix sum of the result can either terminate in the left subrange, or take the left subrange in full plus some elements from the right subrange (prefix, as well). Thus, the maximum prefix sum for the result is a maximum of two values: the left maximum subrange sum and the sum of all elements in the left subrange plus the maximum prefix sum in the right subrange.
3. Symmetrically you can calculate the maximum suffix sum for the created subrange.
4. Finally, the maximum sum of the range can either lie completely in one of the subranges - then we can take the maximum as is, or, the target range can belong the both halves. In this case, it's the sum of the maximum suffix sum from the `LeftOperand` and the maximum prefix sum from the `RightOperand`.

Now the problem is reduced: we've separated the value and the result types, the result one is now more complex. Once you calculate the function over the range, just take the maximum subsegment sum field of the result.

Finally, note that the maximum sum problem is not unique: the type separation generally allows to solve many problems such as:
- The dynamic programming problems with subrange queries, where the state is small enough and two states can be merged. An example for that may be a "House Robber" problem from the LeetCode.
- Although not a dynamic programming problem, the `K`-th order statistics for smaller `K` can also be solved here: the state is the top-`K` values in the subrange.
- The problems requiring the longest subsegment that satisfied the given range. The problem "Hotel" from USACO Gold 2008 can be an example.

### Interfaces

The considerations from the previous sections justify the architecture and allow us to proceed further into the basic interface building. Let's use C++ concepts: they are useful to describe the constraints.

So, we have two types: value type `TValue` and the result type `TOperationResult`. To pass from one to another, we essentially need the type `TOperation` supporting two essential conversions:

- *Repeat*. A method that creates `TOperationResult` from the given `TValue` repeated the given number of times. In case of the range sum, this method is the value multiplied by the count. In case of maximum, it is just the given value. The conventions for the maximum sum case are left as an exercise. Don't hesitate however, to see an implementation in the examples.
- *Merge*. A method that merges two `TOperationResult` instances. Please note that it must be aware of the order, as we target supporting non-commutative functions. For the case of the sum, it's just a sum of two operands. It is a maximum for the case of maximum value. The logic for the maximum subrange sum was described in the previous section.

The design decision here is still to heavily prefer templates over virtual methods, although the latter would probably be simpler. The reason for that is performance: in case of templates, the types are known compile-time, which dramatically improves inlining possibilities. This is clearly not the case with the virtual methods. Note that in real life, the generic implementation is rarely needed: shrinking the template code, where appropriate, is advised.

## Optional operation extensions

The `ValidOperation` concept above is the mandatory minimum: every operation must provide *Repeat* (`FromRepeated`) and *Merge*, and with just those two the engine is already correct and complete. On top of that baseline, the engine recognizes two **optional** capabilities. Neither is required; each, when an operation happens to provide it, unlocks a faster path, and an operation that lacks it simply stays on the generic one. The detection is done with concepts and `if constexpr`, so it costs nothing at runtime and adds no virtual dispatch — an operation that doesn't expose a capability never pays for its absence.

There are five of them, addressing different places where the generic path leaves performance on the table: the *query strategy* on point-update engines (`Undo`), the *leaf work* at the bottom of any engine (`FromMany`), the *bulk in-place modification* a wide point-update engine performs on every update (`ApplyToMany`), and two hooks for the work on a 16-element block that the [wide range engine](#the-wide-engine) and the point engines do: the merge of a *whole* block (`MergeBlock`) and of a *part* of one (`FoldInBlock`). They are independent, and a given operation may expose any subset of them — the integer sum, for instance, exposes all three.

### Undo (HasUndo)

```cpp
template<typename TOperation, typename TOperationResult>
concept HasUndo = requires(const TOperationResult& lhs, const TOperationResult& rhs) {
    { TOperation::template Undo<TOperationResult>(lhs, rhs) } -> std::same_as<TOperationResult>;
};
```

`Undo` is an inverse of `Merge`: given an aggregate and the contribution of one part of it, it recovers the rest. For the sum it is plain subtraction (`Undo(a, b) = a - b`); for xor it is xor again. Many operations have no such inverse at all — you cannot meaningfully "subtract" a maximum, a maximum-subarray state, or a set of order statistics out of an aggregate, since the information needed to do so was discarded by the merge.

When the inverse is available, the single-point engines answer a range query as the difference of two prefixes, `Undo(prefix(r), prefix(l))`, walking one root-to-leaf path instead of assembling the `O(log N)` canonical nodes that tile the range. It is also the hard requirement of the Fenwick-style `TFatPointUpdateEngine`, which keeps running prefixes across wide nodes and therefore must be able to fold deltas in and back out. The point-update factory dispatches on precisely this concept: when `HasUndo` holds it hands back the fat, prefix-based engine; otherwise it falls back to the plain binary tree that tiles the range from canonical nodes. The mechanics of both are detailed in [Single point modifications](#single-point-modifications).

### FromMany (HasFromMany)

```cpp
template<typename TOperation, typename TValue, typename TOperationResult>
concept HasFromMany = requires(const TValue* data, size_t fromIndex, size_t toIndex) {
    { TOperation::template FromMany<TOperationResult>(data, fromIndex, toIndex) } -> std::same_as<TOperationResult>;
};
```

`FromMany` builds a result straight from a contiguous run of raw `TValue`: it receives a pointer and a half-open `[fromIndex, toIndex)` range and returns the aggregate over them in a single call. The generic alternative is to fold the block element by element, merging `FromRepeated(value, 1)` into an accumulator once per element; `FromMany` lets an operation that can summarize a whole run more cheaply do so in one shot — summing a strip of integers is the obvious example. It is consulted exactly where the engine meets a contiguous strip of raw values: at the fat leaves of the range engine, both when rebuilding a leaf during an assignment and when answering a query that ends inside one, and at the brute-force off-cuts of the point engine. Where it is absent, the engine quietly uses the `Merge` loop instead.

#### SIMD discussion

The shape of `FromMany` is an open invitation to SIMD. It is handed a flat, contiguous buffer and a range and asked for a reduction over them — exactly the pattern that maps onto vector instructions: load eight lanes, combine them with one wide instruction, advance, repeat. Reaching for `_mm256_add_epi32` on the integer sum, or `_mm256_max_epi32` on the maximum, looks like the obvious next step, and the interface was deliberately shaped to permit it.

In practice the case for doing it by hand looked hard to justify a priori, because the blocks `FromMany` actually sees are small. A fat leaf spans `LeafSize` elements and a partial leaf fewer, and the whole purpose of the surrounding machinery — cached subtree values, lazy tags, prefix walks — is to *avoid* scanning long runs in the first place. The vectorizable part is therefore a short tail, dominated by call and reduction overhead rather than by steady-state throughput.

We tested it regardless — and the test turned out not to measure what it claimed, which is worth recording. Hand-written AVX2 variants of both the sum and the maximum/minimum operations were added and benchmarked against the scalar versions, with the leaves pushed all the way to `LeafSizeLog = 10` — blocks of 1024 elements — to hand the vector code as much contiguous work as the structure allows. The outcome was no measurable difference at any leaf size, in either engine. The reason, found much later, was not that the blocks are short. The `HasFromMany` concept passes `TOperationResult` as its single explicit template argument, so an operation must declare `FromMany` with the result type *first*; those variants declared it as `<TValue, TOperationResult>`, the argument bound to `TValue` instead, `TOperationResult` was left undeducible, and the concept silently evaluated false. The engine therefore took its scalar `Merge`-loop fallback in **both** arms of the comparison: the two binaries ran the same code. The variants have since been removed, and the question of what an explicitly vectorized `FromMany` buys the engine is open again — the concept mismatch has to be repaired before it can even be asked.

Disassembling the compiled scalar operation explains why. Under the release flags (`-O3 -march=native`) the compiler already auto-vectorizes the `FromMany` reduction on its own. The inner loop of the integer-sum `FromMany` compiles to a 256-bit accumulation followed by a horizontal fold:

```asm
.L4:                                    ; hot loop — one 256-bit accumulator (%ymm1)
        vpaddd  (%rax), %ymm1, %ymm1    ; add 8 int32 lanes at once
        addq    $32, %rax               ; advance by 8 elements (32 bytes)
        cmpq    %rdx, %rax
        jne     .L4
        vmovdqa %xmm1, %xmm0            ; horizontal reduction of the 8 lanes:
        vextracti128 $0x1, %ymm1, %xmm1 ;   fold the high 128 bits into the low,
        vpaddd  %xmm1, %xmm0, %xmm0     ;   then add the halves pairwise,
        vpsrldq $8, %xmm0, %xmm1
        vpaddd  %xmm1, %xmm0, %xmm0
        vpsrldq $4, %xmm0, %xmm1
        vpaddd  %xmm1, %xmm0, %xmm0     ;   down to a single int32
        vmovd   %xmm0, %eax
```

That is the very same `vpaddd` over a `%ymm` register we would have written by hand. The maximum reduces identically, with `vpmaxsd` in place of `vpaddd`. There is simply nothing for hand-written intrinsics to add.

One nuance deserves a precise statement, since it is easy to assume more than is true. The auto-vectorized loop above uses a *single* vector accumulator (`%ymm1`); under the default flags the compiler does **not** unroll the reduction across several independent accumulators to hide the latency of the dependent `vpaddd` chain. A four-accumulator form — produced either by hand-written intrinsics or by adding `-funroll-loops -fvariable-expansion-in-unroller` — is genuinely faster, but only on long, isolated reductions of hundreds to thousands of elements; for the sum that flag alone closes the entire gap, and for the maximum it narrows it. (Notably, hand-unrolling the *source* into four accumulators achieves nothing on its own: the optimizer's reduction recognizer merges them straight back into one.) None of this reaches the engine, for the same reason as before — the blocks are far too short for accumulator-level parallelism to pay off.

(A later round found where hand-written SIMD does pay off: not in long reductions but in removing a mispredicted loop exit on short, variable-length blocks, and in fixed 16-element folds of multi-field states -- see [FoldInBlock](#foldinblock-hasfoldinblock) and [MergeBlock](#mergeblock-hasmergeblock).) The conclusion is that we ship no hand-vectorized `FromMany` for any of the provided operations: the compiler vectorizes the simple reductions for us, as the disassembly above shows. Whether the structure ever feeds them runs long enough for a hand-tuned kernel to earn its keep is, per the paragraph above, still unmeasured. The `HasFromMany` interface leaves the door open in principle — an operation whose block summary is genuinely expensive, or a deployment that deliberately runs very fat leaves, is free to supply a vectorized `FromMany` — but the engine will only pick it up once the concept matches the declaration.

### ApplyToMany (HasApplyToMany)

```cpp
template<typename TOperation, typename TOperationResult>
concept HasApplyToMany = requires(TOperationResult* block, size_t count, size_t from, const TOperationResult& delta) {
    { TOperation::template ApplyToMany<TOperationResult>(block, count, from, delta) };
};
```

`ApplyToMany` folds a single `delta` into a contiguous run of accumulators in place: its contract is exactly the loop `for (i = from; i < count; ++i) block[i] = Merge(block[i], delta)`, with `block[0 .. from)` readable but left untouched. The name describes *what* it does — merge one value into many — deliberately, not *where* the engine happens to need it, so that an operation author need not understand the tree's internal block layout to provide one.

Where the engine needs it is the wide, Fenwick-style point-update engine ([fat engine](#bottlenecks-fat-engine)): on every update it folds the delta into the *suffix* of one sibling block per level, and by default does so with a short scalar loop. An operation that exposes `ApplyToMany` can replace that loop with whatever it likes — the natural choice for a fixed-width block being a single masked vector instruction that touches the whole block at once, masking off the prefix it must not modify. `TSumOperation` and `TBitwiseXorOperation` provide such AVX2 implementations for the integer and `double` widths; an operation that provides none keeps the portable scalar fallback, so the engine stays fully generic. This is the same masked-block idea that the wide segment tree uses for its propagation, exposed as one optional operation method rather than baked into the structure; the [vectorized-update concept](#borrowing-the-wide-trees-trick-a-vectorized-update-concept) section walks through how it came about and what it measures.

### MergeBlock (HasMergeBlock)

```cpp
template<typename TOperation, typename TOperationResult>
concept HasMergeBlock = requires(const TOperationResult* block) {
    { TOperation::template MergeBlock<TOperationResult>(block) } -> std::same_as<TOperationResult>;
};
```

`MergeBlock` merges exactly 16 consecutive results, in order: its contract is `Merge(... Merge(block[0], block[1]) ..., block[15])`. That is the bulk of the work of an assignment in the [wide engine](#the-wide-engine), which refolds one 16-slot node per level on both boundary paths; without the hook it is a chain of 15 `Merge` calls, each waiting for the previous one. A fixed shape invites SIMD, and a fixed shape *only* -- a full block needs no neutral element, so the hook is safe for operations that do not have one.

The maximum-subarray operation implements it with AVX2. Its state is four `int32` fields, `{best, prefix, suffix, sum}`, and all four merged fields are "the larger of a sum and an old field", so one merge is a couple of in-register shuffles, an add, and a max -- for two merges at once, one per 128-bit half of a register. Sixteen states fold as a tree 16 -> 8 -> 4 -> 2 -> 1. The measured effect is in [Hooks for the wide engine](#hooks-for-the-wide-engine).

### FoldInBlock (HasFoldInBlock)

```cpp
template<typename TOperation, typename TOperationResult, typename TElement>
concept HasFoldInBlock = requires(const TElement* block, size_t from, size_t to) {
    { TOperation::template FoldInBlock<TOperationResult>(block, from, to) } -> std::same_as<TOperationResult>;
};
```

`FoldInBlock` merges the elements `[from, to)` of a 16-element block, for `from < to`. `TElement` is what the block holds: results (a node of the wide engine) or raw values (a leaf, or the value array of a point engine); for values the contract is the chain over `FromRepeated(block[i], 1)`. All 16 elements may be read.

The point of the hook is not the arithmetic but a branch. A partial block has a data-dependent length, so a loop over it ends at a different iteration on almost every call, and its exit branch mispredicts -- the very effect [measured in the fat engine](#where-the-generic-engines-query-time-goes-the-fat-leafs-brute-force-scan), where it flushes the out-of-order window holding the next queries' cache misses. An operation with a known neutral element can take the whole block in SIMD lanes, replace the lanes outside `[from, to)` with that element under a mask, and reduce with no branch at all. The sum (neutral 0) and the maximum (neutral `INT32_MIN`) implement it for `int32` with AVX2.

It is deliberately used only for *partial* blocks. A whole block is a loop of fixed length, which the predictor learns and the compiler vectorizes by itself; routing it through the masked kernel measured 7.5% slower on assignments.

### Checking that a hook is there

Like `FromMany` before it (see the [SIMD discussion](#simd-discussion)), a hook whose signature drifts slightly does not fail to compile: the concept just evaluates to false and the engine silently takes the generic path. The check is a `static_assert` on the same concept the engine uses:

```cpp
static_assert(NProust::HasFoldInBlock<NProust::TSumOperation, int32_t, int32_t>);
static_assert(NProust::HasFoldInBlock<NProust::TMaxOperation, int32_t, int32_t>);
static_assert(NProust::HasMergeBlock<NProustExamples::TMaxSumStateOperation, NProustExamples::TMaxSumState>);
```

The shipped hooks exist only when the code is compiled for AVX2 (`-march=native` in the release build, or `-mavx2`); elsewhere these asserts fail by design, and the engines run the plain `Merge` chains. The tests therefore run twice: as usual, and in a second build with `-mavx2`, where the engines' stress tests go through the hooks, and the hooks are also compared directly against the `Merge` chain on every `[from, to)` of random blocks.

## Building a robust engine

We start solving this problem with a segment tree implementation. Please refer to a comprehensive tutorial on the topic if you haven't worked with the segment trees before. Below is a brief explanation.

The tree is built in a way that each node corresponds to an interval and is either a leaf or has two children. The root node corresponds to the full range, that is `[0; N)`. If this range is longer than just one element, it has two children: the one responsible for the range `[0; N / 2)` and the one responsible for the range `[N / 2; N)`, integer division is used. Recursively, one can expand the ranges more and more, until the tree only contains leafs.

Each node then contains two values:
- The value of the target function `Func` for all of the elements that correspond to the range denoted by the node. That is, the root node contains the value of the target function for all of the elements. This is clearly a `TOperationResult` in the terms above.
- Optional: if all of the elements are corresponding to a certain element, the value of this element. This is trickier, and in the general case it is `std::optional<TValue>`.

When an assignment takes place, we descend from the root, targeting the nodes affected by the modification range. Once the modification range covers the node in full, we update the optional described above with an actual value and don't traverse further. The value of the target function is then just the result of the repetition (*Repeat* conversion) of the target value as many times as the number of elements in the range corresponding to the node.

When we go back, we recalculate the value of the function in the parent nodes: the modification has changed this value in children, the parent needs to be updated as well. This uses *Merge* conversion.

If, traversing the nodes during the modification, we see that the optional value is set, we push it forward to the children along with updating the value of the `Func` on the corresponding children ranges. Eventually, the value of the earlier modification, will be delivered to all of the children, but only if and when needed. This technique is called *lazy propagation*.

The query is similar: we start from the root and traverse all of the children. Once we see a node that is fully covered by the query range, we take the value of the target function in this node and exit. The same can be done, if the node is fully covered by a lazily propagated value: it can then just be repeated as many times as the number of elements in the intersection of the target range and the range denoted by this node.

It can be proven that both operations have `O(log N)` complexity. Still, it's rather *slow*, and is only a beginning of the solution.

### Memory management

The first thing that needs to be eliminated, if targeting for nanoseconds, are memory allocations. The structure described above contains `2 * N` nodes, and memory for them must be allocated in advance, making sure no calls to the allocator will be done in between.

Not all pre-allocations are equally good. In order to avoid jumps and reduce cache misses, the allocated memory must essentially be a consecutive chunk of memory. That is, we don't use pointers and connected ways to allocated memory one by one. Equally, we don't use lists. The memory is allocated once, as a `std::vector` that ensures the continuity.

The memory is allocated as a single chunk. The next question is how to use this chunk efficiently.

#### Node topology

Consider a chunk of `2 * N` elements and a need to assign certain places of this chunk to certain nodes of the abstraction described above. There are two similar ways to do that. The first way is to use the binary heap topology: the root is the node numbered `0`. For a node `NodeNumber`, the children are the nodes numbered `NodeNumber * 2 + 1` and `NodeNumber * 2 + 2`. Note that there are also variations where the root node has the index `1`. Then, the children can be computed as `NodeNumber * 2` and `NodeNumber * 2 + 1`.

This is one of the most popular layouts, but it has a caveat: suppose the structure sizes equal to a power of two plus one, for example, `9`, `17`, `33` and so on. Let's calculate the index of the rightmost node for those structures:

- Let's start with a structure of the size of two, it's `2`.
- For a structure of three, the root has the left child corresponding to the range `[0; 1)` and the right child corresponding to the range `[1; 3)` rooting at the node `2`. The right child will then have the index of `2 * 2 + 2 = 6`.
- If we consider a structure of five elements, the story repeats. The right child corresponds to the subtree of the size of three. That means, we end up with `(2 * 2 + 2) * 2 + 2 = 14`.

And so on, the structure for `2^P + 1` elements will have the rightmost leaf corresponding to an index `2^(P+2) - 2`,  which can be proven by induction. We have therefore several problems: this way leads to excessive allocations of memory, which is bad from the resource consumption standpoint.

To mitigate that, we can experiment and number the nodes pre-order. The leftmost child of the node can then be computed just as `NodeNumber + 1`, while to compute the rightmost one, we can make use of the fact that for the structure size `N`, the number of nodes in a full segment tree, equals to `2 * N - 1` (this again, can be proven by induction). We can then pack the tree in a compact way while having `O(1)` formulas.

Both layouts have pros and cons: the binary heap layout is easier to adjust when dismantle nodes that correspond to the small ranges, while the pre-order one can be more memory-efficient if one needs to use the full tree. Both are defined by a concept and are implemented in the engine. Whatever is chosen, the memory must be allocated and the tree must be prepared in advance, before the hot paths.

#### SoA versus AoS

When the numbering is done, there are two ways to place the complex node structure in memory. The first way is to use an array consisting of structures, in our case, pairs of a `TOperationResult` for the node value and the `std::optional<TValue>` for the lazily propagated value. This layout is called an *array of structures*, or simply *AoS*.

*AoS* has several problems, due to which it is often discouraged. The principal problem is *padding* that undermines cache efficiency if the size of the padded structure is bigger that just the sum of the type sizes: the structures are placed in memory consecutively, and we want the cache line to contain as much useful information as possible. Adding the meaningless padding bytes instead of our data is a bad strategy.

To address that, the layout called *structure of arrays*, or simply *SoA* is used. The data is then stored as two arrays: one for the `TOperationResult`, and one for `std::optional<TValue>`. The influence of padding is much smaller since all of the elements in each of the arrays have the same type. It allows to pack the data more tightly and to use the cache lines more efficiently.

Both layouts have been implemented in the engine. You may implement your own: the only requirement is to satisfy the concept. Though both are meticulously tested, the benchmarks rely on *SoA* due to the considerations described above.

#### Packing densely when possible

The SoA layout keeps the lazily propagated values in their own array. The most natural way to represent such a value is the `std::optional<TValue>` mentioned above: the value itself, plus a boolean telling whether anything is actually stored. This boolean is exactly the kind of thing that ruins the packing we've just fought for. A `bool` next to, say, an `int32_t`, forces the structure to be padded to the alignment of the value: the resulting tag is eight bytes where the payload is only four. Across millions of nodes, half of the lazy-tag array is then meaningless padding, and we're back to wasting cache lines.

The observation that saves us is that in practice the value domain is rarely used in full. If there is a value that can never legitimately be assigned, we can reserve it as a *sentinel* and let it stand for "no pending modification". For an `int32_t` sum, `std::numeric_limits<int32_t>::max()` is a natural candidate. The lazy tag then becomes just the bare value: it is defined whenever it differs from the sentinel, and dropping it is nothing more than writing the sentinel back. No boolean, no padding, perfectly dense.

Both variants are implemented and chosen at compile time, behind a single `TLazyTag` interface. The default `TLazyTag<TValue>` keeps the explicit flag (decorated with `[[no_unique_address]]` so the compiler may at least pack it when the surrounding type allows). The specialization parametrized by a sentinel stores nothing but the value. The engine and the benchmark pick between them through the `Sentinel` template parameter, exposed on the command line as `--with-sentinel`.

The trade-off is correctness, and it rests on the user: the sentinel must be genuinely impossible as a real value, otherwise a legitimate assignment would be silently interpreted as "nothing to propagate". When no safe sentinel exists, the boolean variant is always available as a fallback.

### Avoiding branching

Once memory is laid out well, the next enemy is the branch predictor. When an operation costs around fifty nanoseconds, a single mispredicted branch - some fifteen to twenty cycles on a modern core, several nanoseconds - is already a noticeable share of the budget. There are two complementary ways to fight this.

The first is to help the predictor rather than fight it. The hot path is annotated with `[[likely]]` and `[[unlikely]]` so that the compiler lays the common case out as the straight-line fall-through and pushes the rare case out of the way, keeping the instruction cache dense along the path we actually take. For example, an assignment over an empty range is `[[unlikely]]`, and bottoming out at a leaf during the descent is `[[unlikely]]` too: on a large structure, the overwhelming majority of operations terminate at internal nodes long before reaching the leaves.

The second, and the stronger one, is to remove the data-dependent branch altogether and let it become arithmetic. The merge of two maxima is written as a ternary rather than an `if`, so the compiler emits a conditional move instead of a jump; the same holds for the `std::min`/`std::max` clamps used when reconciling a partial leaf. A conditional move has no misprediction penalty because there is nothing to predict.

There is a limit, though, that is intrinsic to the structure. The recursive descent of the segment tree is inherently branchy: at every level we must decide which of the two children intersect the query range, and those decisions depend on data the predictor cannot foresee. Lazy propagation only adds to it, since whether a tag has to be pushed down is, again, a data-dependent branch. This is one of the structural costs of the range engine. As we will see in the [single point modifications](#single-point-modifications) section, dropping lazy tags lets us replace the whole descent with a short, straight-line sequence of bit operations - branchless by construction.

### Being smarter about smaller intervals

A textbook segment tree recurses all the way down to ranges of a single element. This is elegant but wasteful, because the lowest levels of the tree are where almost everything lives: half of the nodes are leaves, a quarter are their parents, and so on. Reaching a size-one range means paying for the deepest, most numerous levels on every single operation.

The remedy is to stop the recursion early. Instead of descending to single elements, we cut the tree off at a *fat leaf* that covers a small block of `LeafSize = 2^LeafSizeLog` consecutive elements, and we keep those raw elements in a flat array next to the tree. Whenever an operation lands inside such a block, it is finished with a tight linear loop over at most `LeafSize` elements - which is cache-friendly, branch-predictable and trivially vectorizable - rather than with another `LeafSizeLog` levels of pointer-free but still index-chasing traversal.

The benefits compound. The tree becomes shallower, so there are fewer levels to descend and fewer merges per operation. There are fewer nodes overall, so less memory is touched: the topology classes reflect this directly, reserving less capacity once the leaves are fat (the pre-order layout drops from `2 * N` down to roughly `N`). The bottom of the structure is now a contiguous run of values instead of a layer of scattered nodes. And when the leaf size is one, the flat array is not even allocated - each leaf node simply holds its own result.

The cost is the linear scan: resolving a partial leaf is `O(LeafSize)`, so making leaves too large slowly turns the structure back into a brute-force scan. There is therefore a sweet spot, and it depends on the operation and the hardware. It is found empirically by sweeping `--leaf-size-log` in the benchmarks, while the stress tests exhaustively cover every small leaf size to make sure the corner cases stay correct.

One more hook helps here. For some operations a whole contiguous block can be summarized faster than by repeated pairwise merges - summing a run of integers is the obvious example. When the operation provides a `FromMany` method (detected through the `HasFromMany` concept), the engine hands the whole block to it; otherwise it falls back to a plain `Merge` loop. This keeps the generic path correct while letting cheap operations shortcut the leaf work.

## Single point modifications

As noted earlier, the point modification is a special case of the range one, with `RangeStart = RangeEnd - 1`. The extra constraint is worth a dedicated engine, because it removes the single most expensive feature of the generic case: lazy propagation. A point assignment touches exactly one leaf and the `O(log N)` nodes on the path from that leaf to the root. There is nothing to defer and nothing to push down, so the whole machinery of lazy tags - and the data-dependent branches that come with it - simply disappears.

`TRobustPointUpdateEngine` is built around this. It uses a binary-heap layout over an array of `2 * AllocatedLength` nodes, where `AllocatedLength` is the input size rounded up to a power of two with `std::bit_ceil`. The leaves occupy the second half of the array, so the element at position `i` lives at node `AllocatedLength + i`. The tree is one-indexed: although a touch less natural, it makes the topology arithmetic cheaper than zero-indexing, since the parent of a node is just `i >> 1` and its children are `i << 1` and `(i << 1) | 1`. An assignment writes the leaf and then walks straight up to the root, recomputing each parent as the merge of its two children in a single iterative, branch-free loop.

Querying splits into two regimes, depending on whether the operation has an inverse.

- When the operation provides an `Undo` (for the sum, `Undo(a, b) = a - b`), a range is computed as the difference of two prefixes: the answer for `[l, r)` is `Undo(prefix(r), prefix(l))`, where each prefix is accumulated by walking a single root-path. This is the cheapest of all, touching one path instead of two subtrees.
- When there is no meaningful inverse - you cannot "subtract" a maximum, a maximum-subarray state, or a set of order statistics - the range has to be assembled from the `O(log N)` canonical tree nodes that exactly tile `[l, r)`. Here lies the payoff promised in the [avoiding branching](#avoiding-branching) section: because there are no lazy tags and the layout is a plain heap, the exact set of covering nodes can be enumerated with cheap bit tricks - isolating the lowest set bit with `x - (x & (x - 1))`, clearing it with `x & (x - 1)` - instead of a recursive descent. The traversal runs forward as far as the greedily largest node allows, then finishes the remainder backwards; the in-code comment reports a fourfold speedup over the recursive version, before SIMD and the fat-leaf optimization are even applied. The two passes are kept separate, and the operands fed to `Merge` in strict left-to-right order, precisely because the operation may be non-commutative.

The fat-leaf trick from the previous section layers cleanly on top of either regime: the aligned core of the range is served by whole tree nodes, while the at-most-`LeafSize` elements hanging off each end are mopped up by a brute-force scan or by `FromMany`.

### Bottlenecks. Fat engine

Even with lazy tags gone, the point engine has a structural bottleneck: depth. Every assignment and every prefix walk costs `log2(N)` steps, and these steps are *serially dependent* - each level reads a node produced from the level below, so the loads form a chain that cannot be overlapped. At a million elements that is roughly twenty dependent memory accesses, and what stalls the pipeline is the latency of that chain, not the arithmetic of the merges. The merge units sit idle waiting for the next load to return.

The way out is to make the tree wider and therefore shorter. `TFatPointUpdateEngine` uses a branching factor of `2^BranchingLog` instead of two; the factory wires it up with `BranchingLog = 4`, that is sixteen children per node. A sixteen-way tree over a million elements is only about five levels deep instead of twenty, so the dependent-load chain shrinks by the same factor.

Width is not free, of course. Recomputing a wide node from sixteen children is more work, and a prefix that ends in the middle of a node would, naively, require scanning its siblings. The "fat" engine pays for this in advance: each node stores a *running prefix* across its siblings, so that a prefix query reads one precomputed value per level rather than scanning a fan of children. This is exactly the idea behind a Fenwick tree, generalized to a wide, cache-conscious node. It does require the operation to be invertible, which is why the engine is constrained to operations that expose `Undo`.

Concretely, an assignment computes the delta between the new and the old value (`Undo(new, old)`), writes the point, and folds that delta into the sibling prefixes along the path to the root. A range query is once again `Undo(prefix(end), prefix(start))`, with a brute-force fast path for ranges too short to be worth the machinery. A small detail in `Init` overbooks the capacity by one element, which lets the backward prefix walk avoid an extra bounds branch.

This split is why the point-update factory dispatches on the operation: when an `Undo` is available it hands back the fat, Fenwick-style engine; otherwise it falls back to the plain binary tree with the branchless node-tiling traversal described above.

## Benchmarking methodology

There are two common variations of the task: the one where the modification is a segment, and the one where the modification is a point. The latter is clearly a subtask of a more generic case, having `RangeStart = RangeEnd - 1`. The extra constraint allows to solve the task more efficiently: in the same `O(log N)` complexity, but with smaller constant factor and better measurements. It therefore makes sense to compare the both.

As for the generic setup, four tasks were chosen:
1. The task where `Func` corresponds to the element sum.
2. The task where `Func` corresponds to the maximim value.
3. The task where `Func` corresponds to the maximum subrange sum.
4. The task where `Func` corresponds to the `K`-th order statistic, we set `K = 16`.

In all of the tasks we put the structure size as `1_000_000` and the number of queries as `10_000_000`.

We target correctness and measurement precision.

To achieve correctness, we first decouple the dataset creation and the dataset execution. A naive engine is used in order to create a dataset for each instance. Note that a naive implementation is much slower: it executes each of the assignments and queries linearly and takes a while to create the dataset. Still, this option is preferred in order to be able to compare two completely different implementations and achieve comparison against a simple and correct implementation.

The measurement precision is harder to achieve on a regular hardware. A few things are done in the benchmarking harness to reduce jitter:
- The execution is pinned to a single core. This way we avoid the effect of cold cache when a context switch task place. The pinning is also necessary to ensure the correctness of RDTSC calls, used for time measurements.
- The first 5% of the dataset is excluded from the analytics. This data serve for the warmup, to soften the effects of a cold start.
- The number of programs running at this time in the OS is reduced to a minimum.

These measurements are not enough: the interrupts can still be handled by the core that executes the benchmarking harness. In addition, even though the process is pinned to a single core, other processes may also occupy this core. To avoid that, a separate OS-level setup with `isolcpus` must be done in the production environment.

Time measurements are approached from two angles.

First of all, we naturally want to use the wall clock, or, to be more exact, monotonic high precision clock in C++ to measure the amount of time spent in the most natural way. This is not ideal: as we will see further, in some of the cases the code runs in around 50 nanosecond per query, moreover, we need a way to evaluate improvements that are around 5-10 per cent of the above. The overheads of the syscall for getting current time will affect the precision of the measurement. We therefore bulk the queries into the batches of several hundred, and measure the time twice: at the beginning of the batch processing and at the end of this processing. The duration, divided by the size of the batch, gives an approximation over a single query length within.

The approach above has clear limits, and the worst part is an impossibility to distinguish, in the mixed datasets, the durations and the percentiles of an assignment and the durations and percentiles of a query. The lightweight solution here is to use RDTSC and measure the execution in TSC ticks.

Without giving all of the important historical details of the TSC (this may take a long while), there are several vital considerations to make:
- The methods that get the number of ticks at the start and at the end of the measurement have to be different, because proper memory and compiler fences must be applied. Thus, for the start of the computation, reading of the counter is preceded by `LFENCE`. Note that in the general case, this instruction does not give enough fencing guarantees for you - it only makes sure that the load operations *before* the fence are executed; however, on Intel CPUs it gives stronger guarantees in practice and can be used here. Note that it is not the same for AMD, and you'll need to use something serializing like `CPUID`. In turn, the finish of the execution is measured with `RDTSCP` that waits for everything to be serialized before.
- The CPU is not the only thing that could initiate rearrangement of the instructions: the compiler could do that as well. In order to do that, we use compiler fences.
- We deliberately don't translate the TSC ticks into nanoseconds, although in some cases it is possible. The reason for that is that there are different types of TSC, and not all of them ensure the independence from C-states, P-states and T-states; without that, it is impossible to ensure a precise translation. The computer used to run the benchmarks has however, invariant TSC that allows the same. Therefore, basic intuitions on that will be given.

**Two measurement modes.** The per-operation `RDTSC` instrumentation is not free, but the cost is subtler than "the fences are slow", and getting it right is what motivates the second mode. The `LFENCE`/`RDTSCP` pair around each operation has two distinct effects. The first is a *fixed* cost — the instructions themselves — and it is small: timed in isolation (fences around a no-op) it comes to about **21 ns** on this machine, tens of nanoseconds, and the same for every engine. That fixed cost alone cannot explain a per-engine spread of hundreds of nanoseconds, and it doesn't. The second effect is the one that matters: the fences are *serializing*, so each operation must fully retire before the next begins. That does not add an overhead so much as it changes *what is measured*. A bare loop measures **throughput** — the CPU overlapping one operation's cache misses with the next's; a fenced loop measures **latency** — one operation start-to-finish, alone. For an operation that is mostly independent memory accesses those two numbers diverge enormously, and the gap is a property of the algorithm, not of `RDTSC`.

A small experiment (`benchmarks/fence_experiment.cpp`) makes it concrete — the prefix-sum query timed in a bare loop versus a fenced loop:

| engine | throughput (bare) | latency (fenced) |
| --- | ---: | ---: |
| robust (fat) | ~15 ns/op | ~47 ns/op |
| fenwick | ~13 ns/op | ~48 ns/op |
| wide tree | ~5 ns/op | ~48 ns/op |

Fenced, all three converge to ~47 ns — essentially one serialized cache miss plus the ~21 ns fence floor — so per-operation timing makes them look *alike*. Unfenced, they spread apart: the wide tree's query is little more than a handful of independent loads, so the out-of-order engine keeps roughly ten of them in flight and throughput falls to ~5 ns, while the engines whose queries carry data-dependent work and branches overlap less and stay near ~14 ns. Per-operation timing therefore flatters the dependency-bound engines and hides the wide tree's real advantage; only a throughput measurement exposes it. The runner accordingly offers both, chosen with `--measurement-mode`:

- `per-operation` (the default): every operation is bracketed by the fences, yielding the per-operation TSC histograms and the modify/query split. It reports *latency*. Use it when operations are individually expensive — so neither the fixed cost nor the lost overlap matters much — or when the per-operation distribution is what you want.
- `average`: the instrumentation is dropped entirely; whole batches are timed with the wall clock and divided by the operation count. It reports *throughput*, operations free to overlap — at the price of the per-operation breakdown, which is not collected (its TSC columns are reported as zero).

The choice is not academic. Replaying the mixed int32 point-sum dataset, the wide tree measures slower than the generic engine in `per-operation` mode (≈143 vs ≈117 ns/op — its memory-heavy updates have the longest serialized latency) yet clearly *faster* in `average` mode (≈35 vs ≈65 ns/op, with Fenwick at ≈49) — the same code, answering two different questions. For the genuinely expensive range engine, whose operations cost hundreds of nanoseconds and overlap little, the two modes agree to within noise. And this is exactly why the average mode matters for the comparison below: the specialized prefix-sum structures spend almost all their time on overlappable memory accesses, so latency-style per-operation timing is the wrong instrument for them, and only throughput gives them a fair hearing. Rule of thumb: `per-operation` when the operation is expensive or you need percentiles; `average` when it is cheap and heavily pipelined.

**Measuring a universal engine next to specialized reference implementations.** The engine here is deliberately universal: the value type and the result type may differ, and any associative operation with the right hooks plugs in. A natural question for any such generic structure is what that generality costs relative to code written for exactly one problem — and the most instructive way to find out is to put it next to a couple of well-understood specialized implementations and read the difference honestly. For the one partial case that has them — point modifications plus prefix/range sums over an `int32` array — we use two references (see [External reference implementations](#external-reference-implementations-sumint32-point)): a handwritten Fenwick tree, the textbook answer, and the **wide segment tree** from Sergey Slotin's [Algorithmica/HPC](https://en.algorithmica.org/hpc/data-structures/segment-trees/) (I would like to thank Sergey Slotin for the idea behind it). It is a wide (`B = 16`) tree laid out as a flat array of cache-line-sized nodes holding running prefix sums, and it propagates a point update with a single branchless **AVX2** masked add per level — SIMD doing in one instruction the propagation that a scalar tree performs one node at a time. A specialized structure that solves a narrower problem should of course be faster on that problem; the point of putting the generic engine beside it is not a contest but a calibration — to see *where* the cost of generality actually lands, and (as it turns out) to borrow one of its ideas. There is a measurement subtlety that has to be got right first, though, or the comparison says the opposite of the truth.

One setup detail earns its own mention, because it turned out to matter. The engines were originally driven through the `IPointUpdateEngine` / `IRangeUpdateEngine` virtual interface — one indirect call per operation. Holding each engine by its concrete type instead, so the hot-path calls devirtualize and inline (the implementations are marked `final` to guarantee it), improved the generic engine's own query latency by roughly 15–30%, and the wide tree's by about 70%. Virtual dispatch, in other words, was not free even for our work-bound engine, and was outright costly for the latency-bound wide tree. That devirtualization is unconditional and was kept; the per-operation-versus-average decision above is the part deliberately left to the caller.

## Benchmark results

There are two ways to time the same replay, and they answer genuinely different questions (the [methodology](#benchmarking-methodology) derives the distinction in full; here is the practical summary). One can bracket **every individual operation** in serializing fences and time it on its own. That gives the finest possible granularity — a full latency histogram down to a per-operation measurement, and a clean split of modifications from queries — at the cost that the fences forbid the processor from starting the next operation before the current one retires, so what is measured is *isolated latency*. Alternatively one can time the replay **in aggregate** — whole batches on the wall clock, no fences. That gives up all per-operation detail, but it reports what a production stream actually achieves, including the cases where the out-of-order core starts one computation without waiting for a previous one's result (memory-level parallelism). Neither mode is "more correct": they describe the cheap, pipelined operations and the expensive, isolated ones quite differently, so both tables are given below.

Both use the same **optimized** engine — after the changes in [A round of empirical optimization](#a-round-of-empirical-optimization), superseding the pre-optimization snapshot in [`benchmark_report.md`](benchmark_report.md) — on the same eight datasets (four tasks, each in a point and a range flavor) with the same configuration: the `robust` engine, `SoA` layout, binary-heap topology, leaf-size-log 4, no sentinel. Each cell is the median of five runs. The structure length is 10⁶ throughout; the query stream is 2·10⁶ operations for the cheap sum and max tasks and 10⁶ for the heavier max-subarray and order-statistic ones, where the naive ground-truth generation is far slower. The runner records the processor in every CSV row and prints it on startup, so a result is never separated from the hardware that produced it.

**Processor.** `11th Gen Intel(R) Core(TM) i5-1130G7 @ 1.10 GHz` — a four-core / eight-thread, 15 W Tiger Lake part with a 1.10 GHz base and a ~4.0 GHz single-core turbo, and an 8 MB L3. The benchmarks were run without the `isolcpus` isolation the methodology section recommends, so treat the absolute latencies as indicative rather than definitive, and the *relative* comparisons between tasks and modification flavors as the real takeaway. Latencies above 1,000 ns are reported in microseconds (`us`).

### Per-operation latency (barrier-separated)

The fenced `per-operation` mode: every operation is timed alone, which is what yields the percentile columns and the modify/query TSC split. The "Avg latency" column here is the mean of those *isolated* per-operation times — read it as latency, and compare it against the [aggregate throughput](#aggregate-throughput-no-barriers) table below to see how much overlap each operation leaves on the table.

| Task | Modification | Avg latency | Wall p50 | Wall p99 | Wall p99.9 | Peak memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Sum | point | 95 ns | 76 ns | 228 ns | 507 ns | 11.4 MB |
| Sum | range | 254 ns | 129 ns | 313 ns | 504 ns | 12.5 MB |
| Max | point | 96 ns | 85 ns | 239 ns | 521 ns | 11.5 MB |
| Max | range | 261 ns | 131 ns | 316 ns | 472 ns | 12.7 MB |
| Max-subarray sum | point | 204 ns | 180 ns | 484 ns | 815 ns | 13.2 MB |
| Max-subarray sum | range | 348 ns | 160 ns | 395 ns | 554 ns | 14.0 MB |
| Order statistic (K=16) | point | 1.78 us | 1.76 us | 2.49 us | 3.42 us | 19.7 MB |
| Order statistic (K=16) | range | 1.26 us | 400 ns | 1.46 us | 2.60 us | 20.7 MB |

A few things stand out:

- **Point modifications are the cheaper subcase, as predicted — except where state dominates.** For sum, max and max-subarray the point flavor is markedly cheaper on average latency (95–204 ns versus 254–348 ns), confirming the smaller constant factor of the `RangeStart = RangeEnd - 1` specialization. The order statistic inverts it: its point flavor is *slower* (1.78 us versus 1.26 us on average — and 1.76 vs 0.40 us at p50), and at near-identical memory (19.7 vs 20.7 MB) the engine and its working set cannot be the explanation. As the controlled experiment immediately below shows, the gap is almost entirely the **workload**, not the engine.
- **The TSC split shows where the time goes.** Separating modification from query cycles (the reason for using RDTSC at all) reveals that range assignments are modification-bound: for sum the median assignment costs ~628 TSC cycles against only ~129 for a query, because writing a half-open interval has to push lazy tags down `O(log N)` nodes while a query merges a comparable number of already-summarized subtrees. Point modifications, by contrast, are balanced (sum: ~100 modify versus ~79 query cycles).
- **The cost grows with the result type, not just the algorithm.** Sum and max carry a single integer per node and stay around a hundred nanoseconds (point) to a couple hundred (range); max-subarray's four-field state lifts that to ~200–350 ns; the sixteen-element order-statistic state lands in the microseconds. The merge work and the bytes moved per node track the state size directly.
- **Tail latency is environmental.** Across every cell the medians are stable, but the deep-percentile figures (p99.9) sit several times above the median — high-hundreds of nanoseconds for the light tasks, a few microseconds for the order statistic. This is the OS jitter the methodology section warns about: interrupts still land on the pinned core without `isolcpus`, and dataset I/O adds its own noise. The tails describe the machine, not the engine.

### Aggregate throughput (no barriers)

The `average` mode: no fences, whole batches timed on the wall clock and divided by the operation count. There is no per-operation breakdown to report — the percentile and modify/query columns simply do not exist here — but the single number is the one a throughput-bound deployment actually sees, with the core free to overlap independent work across consecutive operations.

| Task | Modification | Throughput | Peak memory |
| --- | --- | ---: | ---: |
| Sum | point | 28 ns | 11.3 MB |
| Sum | range | 216 ns | 12.4 MB |
| Max | point | 48 ns | 11.5 MB |
| Max | range | 215 ns | 12.4 MB |
| Max-subarray sum | point | 161 ns | 13.0 MB |
| Max-subarray sum | range | 300 ns | 13.8 MB |
| Order statistic (K=16) | point | 1.42 us | 19.5 MB |
| Order statistic (K=16) | range | 743 ns | 20.3 MB |

Reading it against the per-operation table makes the methodology's central point concrete. The **cheap point operations gain the most** from dropping the fences: sum-point falls from ~95 ns of isolated latency to ~28 ns of throughput, max-point from ~96 to ~48 — these are short, mostly-independent walks, so the out-of-order core keeps several in flight and the effective per-operation cost is a fraction of the standalone one. The **range assignments barely move** (sum-range 254 → 216, max-range 261 → 215): each carries real, data-dependent work — pushing lazy tags down two fringes — that does not overlap as freely, so latency and throughput nearly coincide. The order statistic sits in between, its heavy `TopK<16>` merges giving the pipeline less to hide. The takeaway is the one the methodology states up front: quote *latency* when an operation is individually expensive or you need its distribution, and *throughput* when it is cheap and runs in a pipelined stream — and never silently mix the two.

### Order statistic: the gap is the workload, not the engine

The order-statistic point flavor looks ~4× slower than the range flavor above (1.76 vs 0.40 µs at p50), and the TSC split localizes it sharply: the **query** is roughly an order of magnitude slower on the point flavor, while modification is within ~1.2×. But the point and range cells differ in *both* the engine *and* the dataset, so neither alone is implicated yet. To separate them, replay **one dataset on both engines**: the range engine reads each point update as `SetRange(pos, pos+1, v)` — identical in effect — and still validates against the same ground truth, so the comparison is honest. Median of 3 runs, TSC cycles for the modify/query columns:

| engine / dataset | avg ns | modify p50 | query p50 | query p95 | query p99.9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| point engine / **point** data | 1546 | 2630 | 2501 | 3777 | 30867 |
| range engine / **point** data | 1830 | 2994 | 3131 | 4598 | 31635 |
| range engine / **range** data | 766 | 2224 | **237** | 595 | 1479 |

Two readings settle it. **Hold the engine fixed and change the data**: the range engine's median query jumps from **237 cycles on the range dataset to 3,131 on the point dataset — ~13×**, essentially the whole gap. **Hold the data fixed (point dataset)** and the two engines' queries are within ~25% — the point engine is, if anything, slightly *faster*. So the engine is not what slows its queries — at near-equal memory (19.7 vs 20.7 MB) the data, not the structure, sets the query cost. (Modification, meanwhile, is workload-independent and roughly equal everywhere — it is engine-bound.)

What actually happens is a property of the **values the modifications leave behind**. Range-assignments paint long uniform runs, and an order-statistic query landing in a uniform run is answered from a single lazy tag — the K-th largest of a repeated value is that value, with no descent and no `TopK<16>` merges, a few hundred cycles. Point updates leave every element distinct, so every query must walk the tree and merge real top-K states down the path — thousands of cycles. Same query code, ~13× the work, purely because of the data present. The general lesson: for a state-heavy operation, the *modification pattern* preceding the queries can dominate query cost far more than the choice of engine does — and a benchmark that pairs each engine with its "natural" workload (point engine ↔ point updates, range engine ↔ range assignments) can mistake that workload effect for an engine difference.

## External reference implementations (SumInt32, point)

A generic engine is best understood next to specialized implementations of the same problem — they make the cost of generality concrete and, here, taught us a technique worth borrowing. For the one partial case that has well-known specialized solutions — point modifications plus range sums over an `int32` array — `include/engine/external/` holds two of them, selectable in the runner via `--implementation`:

- **`fenwick`** (`external/fenwick.h`) — a textbook Fenwick tree (binary indexed tree), the classic answer to point-update / prefix-sum. A range sum is `prefix(r) - prefix(l)`.
- **`algorithmica`** (`external/algorithmica.h`) — a port of the **wide segment tree**, the fastest structure in Sergey Slotin's [Algorithmica/HPC segment-tree chapter](https://en.algorithmica.org/hpc/data-structures/segment-trees/) (file `segtree/simd.cc` in [sslotin/amh-code](https://github.com/sslotin/amh-code/tree/main/segtree)). It is a B=16-ary tree of cache-line-sized nodes storing running prefix sums; a prefix query is one read per level, and an update is one branchless AVX2 masked add per level. The only adaptation is ownership and a slot count padded up to the next power of B, so the whole-array query `sum(length)` stays in bounds.

Both are fixed to the 10⁶ dataset length, accumulate in `int32` to match the ground truth bit-for-bit, and are validated against the naive engine in `tests/test_external_fenwick.cpp` and `tests/test_external_algorithmica.cpp`. The comparison engine ("best") is what the robust factory selects for sum + point: since `TSumOperation` has `Undo`, that is the `TFatPointUpdateEngine` (branching 4), not the plain segment tree.

**The wide segment tree is the fastest of the three — and measuring that correctly is itself instructive.** The main `runner` reports all three within ~25% of each other (≈115–145 ns/op), with the robust engine nominally ahead. That ranking is an *artifact of measurement*, not reality, for two compounding reasons: the runner used to hold the engine through the `IPointUpdateEngine` **virtual interface** (a per-op indirect call), and — inside the same wall-clock-timed batch — it brackets every operation in `StartRDTSC`/`EndRDTSC`, i.e. `LFENCE` + `RDTSCP` serializing **fences**. The wide tree's query is latency-bound — a few scattered reads, almost no arithmetic — so its speed depends on the CPU overlapping consecutive queries' cache misses (memory-level parallelism), and both of those barriers destroy that overlap.

The first of those two is now fixed: the runner holds each engine by its **concrete** type (`RunDataset` builds it and threads it through a `RunBenchmarkLoop` templated on the concrete engine; the engine classes are marked `final`), so the hot-path calls devirtualize. That alone moves the wide tree's measured query from ~233 to ~186 TSC — real, but small — while the wall-clock average barely budges. The reason is what the [methodology](#benchmarking-methodology) lays out: per-operation timing measures serialized *latency*, not *throughput*. The fixed fence cost is only ~21 ns; the rest is the cross-query overlap the `RDTSCP` forbids by forcing each op to retire before the next issues. Measured that way the wide tree's prefix query is ~48 ns (one serialized cache miss), an order of magnitude above its overlapped throughput of ~5 ns — and at that point it is indistinguishable from Fenwick and the generic engine, which serialize to about the same latency. So the full fix needs the other half too — batch-only timing instead of per-op RDTSC — which is exactly the runner's `average` [measurement mode](#benchmarking-methodology): drop the fences, time whole batches, divide by the count. In that mode the wide tree measures ≈35 ns/op, against Fenwick's ≈49 and the generic engine's ≈65 — the ordering one would expect once the measurement no longer throttles it. Devirtualizing was the cheap, unconditional win and is always on; the `average` mode is the deliberate second half, kept opt-in because the per-operation TSC split it gives up is what the project's own engines are tuned against.

A tight-loop microbenchmark (`benchmarks/external_microbench.cpp`, no fences, no streaming, every engine driven through its **concrete type** — no virtual dispatch, exactly as the runner holds them) shows the true picture. Median ns/op over 5M ops, length 10⁶, identical workload for every contender:

| Engine | update | range `[l,r)` | prefix `[0,r)` |
| --- | ---: | ---: | ---: |
| robust fat (best) | 56 | 29 | 16 |
| fenwick | 27 | 23 | 11 |
| **algorithmica (wide tree)** | **22** | **10** | **5** |
| raw upstream `simd.cc` (free functions) | 24 | 8 | 4 |

The wide tree is the fastest structure by a wide margin: its prefix query (~5 ns) and range query (~10 ns) land right next to the raw upstream `simd.cc` (4 ns / 8 ns), which validates the port — ≈2× Fenwick and ≈2.5–3× the robust engine on queries, while on updates it sits at the front of the fast field (~22 ns) and the robust fat engine, at ~56 ns, is the slowest. So the article's claim holds on this very machine; the earlier conclusion that the wide tree "comes last" was a benchmarking error — an artifact of the per-op virtual call and the RDTSC fences the runner used to impose (see above), both of which throttled exactly the cross-query memory-level parallelism the wide tree lives on.

The lesson is about the engine's *architecture*, not the wide tree: a polymorphic, one-call-per-operation interface is the right abstraction for the project's own engines (whose per-op cost is large enough to absorb it), but it imposes a hard floor that a latency-bound SIMD structure cannot reach. Extracting that structure's speed requires either devirtualizing the hot path or a batched query API that keeps many independent reads in flight — neither of which the current `IPointUpdateEngine` offers. The external references thus do more than validate the engine: they mark exactly where its abstraction starts to cost.

Crucially, that cost is *specific to the wide tree*, which is what justifies keeping the polymorphic interface available rather than condemning it — and why the table above needs only one (concrete) row per engine. The same devirtualized-vs-virtual comparison, run on the robust fat engine and on Fenwick, moved them by ~15% at most (and nothing outside the noise on update or range) — only the wide tree swung 3–6×. The reason is what makes the virtual call expensive in the first place: it is a barrier to issuing the *next* query's loads early, so it only bites operations whose entire budget is cross-call memory-level parallelism. The wide tree's query — about five scattered reads and almost no arithmetic — is exactly that; the robust fat engine's query is work-bound (a tree descent with real per-node merges and dependent loads, tens of nanoseconds of genuine work), so an indirect call is a rounding error against it and there is no idle memory parallelism to reclaim. The polymorphic interface is therefore effectively free for the engines it was designed for, and a ceiling only for a structure that has optimized its per-query work down to nothing.

### Where the generic engine's query time goes: the fat leaf's brute-force scan

The table above leaves the robust fat engine the slowest of the three on bare queries, and a separate experiment (`benchmarks/query_microbench.cpp`, range queries on a populated structure) pins down why — it is almost entirely one thing. The `TFatPointUpdateEngine` stores fat leaves of `LeafSize` raw values and answers the partial-leaf part of a prefix query by **scanning** them (`ComputeRangeBruteForce`). That scan touches little memory — at most one cache line of contiguous `int32` — but it is a *data-dependent, variable-length loop*: it runs `prefix mod LeafSize` iterations, a different count on every query. To isolate its cost the benchmark adds a variant that changes **exactly one thing** — it precomputes the intra-block running prefix sums in a parallel array (maintained on every update), so the partial-leaf contribution becomes a **single read** instead of a scan. Everything else is identical: same `std::vector` storage, same recurrence walk, same runtime sizing.

| working set | engine | update | bare query | fenced query | query overlap |
| --- | --- | ---: | ---: | ---: | ---: |
| 10⁶ | robust (orig) | 59 ns | 29 ns | 63 ns | 2.2× |
| 10⁶ | robust + leaf-prefix | 86 ns | **8 ns** | 50 ns | **6.3×** |
| 16·10⁶ | robust (orig) | 137 ns | 80 ns | 208 ns | 2.6× |
| 16·10⁶ | robust + leaf-prefix | 205 ns | **47 ns** | 174 ns | **3.8×** |

The effect on the **unfenced** query is dramatic — 29 → 8 ns at 10⁶, into the wide tree's league — while the **fenced** query barely moves (63 → 50 ns). That split is the whole story, and the [methodology](#benchmarking-methodology)'s bare-versus-fenced distinction explains it. Fenced measures one query alone, and there the scan is worth only its ~one cache line of serialized work (a ~10–25% difference). Bare measures *overlapped* queries, and there the scan was catastrophic — not for the memory it touches but because the variable-length loop **mispredicts its exit branch on nearly every query**, and each mispredict flushes the out-of-order window that was holding the *next* queries' cache misses in flight. The overlap the structure can sustain collapses from ~6× to ~2.2×. Replace the loop with one fixed, predictable load and the query becomes a branch-free sequence of independent accesses the CPU can run ahead through; consecutive queries' misses overlap again, and the overlap springs back to ~6×. **The cause is the memory access pattern — predictable and branchless — not the amount of memory touched.**

This is neither free nor on by default. The fat leaf exists precisely to keep updates cheap: a point update touches only the node tree, never the leaf interior. Precomputing the intra-block prefixes reverses that bargain — every update must now also fix up the suffix of its leaf block — and the table shows the bill: updates slow by ~45% (59 → 86 ns). So leaf-tail removal **trades update cost for query throughput**, paying off only on read-heavy workloads (here roughly more than ~1.3 queries per update). It is exactly the kind of thing that belongs behind a policy flag rather than in the default. The wide tree sidesteps the trade entirely — it maintains the same intra-block prefix sums, but with a single branchless AVX2 masked add per level, so its update stays cheap (~22 ns) — but only by fixing the type to `int32` and the operation to addition, which the generic engine cannot.

### Fenwick versus the generic engine: a cache-miss story

The 10⁶ microbench above puts the textbook Fenwick tree modestly ahead of the generic engine. That ranking **inverts** as the data outgrows the cache. At 10⁶ the whole structure — a few megabytes — fits comfortably in the 8 MB L3, so a query hits almost no cold memory; its cost is dominated by cheap, hot accesses, and Fenwick's leaner per-query work makes it a touch faster. But Fenwick answers a prefix query by walking `popcount(k)` scattered indices — on the order of a dozen at this scale, and growing with `log N` — whereas the B=16 fat tree walks only `log₁₆ N` levels, a much shorter and shallower path. Once the array no longer fits in cache, every one of those accesses becomes a cold miss, and the access *count* starts to dominate.

| working set | fenwick (specialized) | robust generic engine |
| --- | ---: | ---: |
| 10⁶ (in L3) | 25 ns | 29 ns |
| 20·10⁶ (80 MB, ~10× L3) | 125 ns | **64 ns** |

At 20 million entries the **generic engine is ~2× faster than the hand-tuned Fenwick tree**, purely because its wider branching gives it fewer levels and therefore fewer cache misses per query. The fenced numbers confirm the mechanism: Fenwick's serialized per-query latency grows ~4× from 10⁶ to 20·10⁶ while the fat tree's grows ~3×, tracking their different miss counts. It is a small but pointed result — a *universal* structure overtaking a *specialized* one at scale, not in spite of its generality but because the shape it happens to use (a wide, shallow tree) has better memory behaviour than the binary indexed tree. The lesson mirrors the rest of this section: once operations are this cheap, the benchmark is really measuring the memory hierarchy, and depth — how many misses a query must serialize through — is what wins.

## A round of empirical optimization

Everything above describes the engine's design. This section records a later pass whose only mandate was to *find what was left on the table* — to treat the existing code as a subject, profile it, form hypotheses, and keep only the changes a measurement could defend. It is deliberately a log of the process, negatives included, because in low-latency work the discarded hypotheses carry as much information as the kept ones.

### Measuring on a thermally noisy laptop

The benchmarks run on a fan-cooled 15 W laptop, not an isolated server, and the first finding was about the measurement itself. Running five iterations of variant A as a block and then five of variant B as a block — the obvious protocol — was shown to be unsafe: the same binary on the same dataset drifted from a 50 ns/op median to 63 ns/op over a few minutes as the part warmed and clocked down. A sequential A-then-B comparison silently attributes that drift to the code.

The fix is to **interleave**: run A, B, A, B, … round-robin, so the slow thermal drift acts on both variants equally and cancels in the ratio of medians. Every speedup quoted below comes from interleaved A/B over a fixed dataset, comparing two separately compiled binaries, with the per-operation correctness check (every query validated against the naive ground truth) left on throughout. Where a difference is within run-to-run noise it is reported as noise, not as a result.

### Defaults left on the table

Profiling first localized the cost. The range engine's `SetRange` is roughly **eight times** the cost of a `ComputeRange` (≈1000 vs ≈123 TSC cycles at the median): a query stops at the highest already-summarized nodes and only reads, while an assignment writes its way down two fringes and pushes lazy tags into both children at every level. The per-operation cost also grows *faster* than tree depth as the structure outgrows cache, which is the signature of a memory-bound traversal. The levers that matter are therefore the ones that shrink the working set or improve locality — exactly the [sentinel](#packing-densely-when-possible), [fat-leaf](#being-smarter-about-smaller-intervals) and [topology](#node-topology) knobs already in the engine — and the finding was that the *default* configuration used none of them. All three defaults have since been changed to the settings this section arrived at:

- **Topology default.** `CreateRobustRangeUpdateEngine` defaulted its topology to the binary heap, but the `TRobustRangeUpdateEngine` class template defaulted to pre-order — an inconsistency that handed a direct instantiation the slower layout. Interleaved, the heap is ~21–25 % faster on the hot path (sibling co-location: a node's children are adjacent, so the merge-up reads one cache line instead of a scattered one). The class-template default was aligned to the heap.
- **Dense lazy tags.** The sentinel-packed lazy tag (4 bytes instead of a padded `{value, bool}` at 8) is ~16 % faster on range-sum and cuts the node memory by a third; it is correct whenever the value domain reserves an impossible value, which every shipped dataset (`int32` payloads) does.
- **Leaf size tracks the operation's merge cost.** Sweeping `--leaf-size-log` confirmed the sweet spot is real and, more usefully, that it *moves with the operation*. A cheap merge (sum, max) wants large leaves (`leaf-size-log = 6…8`); a heavy merge (the four-field max-subarray state, the sixteen-way order-statistic merge) wants moderate ones (`= 4`), because the partial-leaf scan costs `O(LeafSize · merge)`. A localized, small-range workload pushes the optimum larger still, since a query that fits inside one leaf skips the descent entirely. Across operations and workloads the then-default `leaf-size-log = 0` was consistently the worst choice; tuned, the engine runs ~1.3–2.3× faster on the same data. The default is now `leaf-size-log = 5` — a 32-element leaf, which a later sweep over every shipped operation and both wide and narrow query shapes found to be the best or within noise of the best everywhere. The point engines' leaf width defaults to 4 for the same reason.

A new generator flag, `--max-range-width`, was added to exercise the localized small-range regime the original uniform generator never produced — the realistic shape for a latency-sensitive service, and the one where the leaf-size effect is largest.

### Reading the disassembly

The configuration findings are tuning. Two genuine code defects (a third, specific to floating-point value types, is covered [below](#carrying-it-to-floating-point-and-xor)) were found only by disassembling the hot paths and comparing the template instantiations against each other.

**A stray AVX prologue in the recursive descent.** `ModifyNodes` and `QueryNodes` are the recursion, and for every operation their prologue carried a 32-byte dynamic stack realignment (`and $-32, %rsp`) and a `vzeroupper` in the body. The cause: the leaf-handling code — `FromMany` for the summable operations, an auto-vectorized `Merge`-fold for max — was inlined straight into the recursive function, dragging `ymm` usage and forcing GCC to realign the frame *on every level of the descent*. The bodies were bloated to match (`ModifyNodes` was 537 instructions). At `leaf-size-log = 0` this leaf code is not merely heavy but *dead* — a size-one node is always fully covered before the leaf branch is reachable — yet it was emitted inline regardless. Extracting both leaf paths into `[[gnu::noinline]]` helpers (`ModifyLeaf`, `QueryLeaf`) keeps the recursion free of AVX: `QueryNodes` fell from 143 to 78 instructions and `ModifyNodes` from 537 to 164, the realignment and `vzeroupper` gone. Measured, it is a **7–15 % speedup** on the cheap operations (sum, max), shrinking to noise on the state-heavy ones where the merge dominates and the prologue is a rounding error.

**Loop-control overhead in the point update.** The suffix loop in `TFatPointUpdateEngine::SetPoint` — the one that folds a delta into a node's sibling prefixes — compiled to six instructions per iteration, three of which re-derived the group-boundary test `(index & MASK) != LEFTMOST` for a single useful add. An earlier attempt to replace it with a counted loop had *regressed* by ~4 %, and the disassembly explained why: the counted form let the vectorizer kick in on a loop that trips at most fifteen times, and the head/tail overhead of vectorizing such a short reduction cost more than it saved. The correct fix is a counted loop *that stays scalar* — a precomputed end bound for the four-instruction pointer loop, with vectorization explicitly disabled on the method (`no-tree-vectorize`, guarded to GCC). That nets a **~6.6 % speedup** on point-sum, and corrects the earlier negative: the counted loop was never the problem, its accidental vectorization was.

### What did not work

Several plausible ideas were tested and rejected, each consistent with the same conclusion — the engine is memory-latency-bound on a binary-tree descent, and micro-optimizations that do not change that latency do not move the needle:

- **Software prefetch** of the descent's children regressed by ~3 %: the work per node (a handful of instructions) is far too little to hide a cache miss, so the prefetch only adds instructions.
- **`madvise(MADV_HUGEPAGE)`** on the node arrays — aimed at the TLB pressure of a ~24 MB random-access working set — produced no change and could not even be confirmed to take effect under WSL2 (anonymous huge pages never materialized); the result is inconclusive on this platform rather than negative in general.
- **An `Undo`-based incremental leaf rebuild**, recomputing a partially-overwritten fat leaf in `O(slice)` instead of `O(leaf)`, showed no measurable gain: the auto-vectorized full-leaf `FromMany` is already cheap enough that the leaf reduction is never the bottleneck — the same conclusion the [SIMD discussion](#simd-discussion) reached from the other direction.
- **Removing the `[[unlikely]]` leaf hint** for fat-leaf builds was ~7 % *slower*, confirming the hint is correctly placed even when leaves are large: internal-node visits still dominate per operation.
- **`-funroll-loops` on the runner** and **raising the point engine's branching factor** above sixteen both came out as noise: the existing choices were already at their optima.

### Before and after

The net effect of the kept code changes (heap default, the `noinline` leaf extraction, the scalar counted `SetPoint` loop, and the `always_inline` operation primitives), measured original-versus-optimized on the main datasets. Interleaved, five runs per binary, median reported; configuration held fixed across the pair so the numbers isolate the code change (the heap-default win is separate, since the runner always names a topology explicitly):

| Task / workload | leaf | original | optimized | speedup |
| --- | ---: | ---: | ---: | ---: |
| sum, range, 10⁶ | 0 | 343 ns | 313 ns | **1.10×** |
| sum, range, 10⁶ | 6 | 219 ns | 202 ns | **1.08×** |
| sum, range, small (width ≤ 64) | 8 | 150 ns | 141 ns | **1.06×** |
| max, range, 10⁶ | 0 | 409 ns | 352 ns | **1.16×** |
| max-subarray, range, small | 4 | 264 ns | 244 ns | **1.08×** |
| order statistic (K=16), range, small | 4 | 996 ns | 981 ns | 1.02× |
| sum, **point**, 10⁶ | — | 52 ns | 48 ns | **1.08×** |

The pattern is the one the disassembly predicted: a uniform ~6–16 % on the operations whose per-node work is small enough for a stray AVX prologue or three redundant loop instructions to be a real fraction of the budget, fading to within noise on the order-statistic engine, where a sixteen-way merge dwarfs everything the changes touched. All three changes are unconditional, carry no API change, and leave the full stress suite (50 tests, sanitized) green.

### Carrying it to floating point and XOR

Everything above was found and tuned on `int32` payloads. Repeating the procedure for **`double`** (under both sum and max) and for **XOR** added three dataset types (`SumDouble`, `MaxDouble`, `XorInt32`) and surfaced one defect specific to floating point — again only visible in the disassembly.

Comparing the `int32` and `double` instantiations of the same engine side by side, the integer operation primitives (`Merge`, `FromRepeated`, `Undo`) inline as expected, but their `double` counterparts do not: at `-O3`, and through IPO, GCC emits them out of line and *calls* them. The point engine's per-element suffix update carried a `call` to perform a single `vaddsd`, wrapped in a stack-protector canary and register spills — the runner binary held seventeen such call sites. Nothing about it is semantic (the integer versions prove the bodies are trivially inlinable); it is purely the inliner's cost model declining the floating-point shape. A `[[gnu::always_inline]]` wrapper (`PROUST_FORCE_INLINE`, guarded to GCC/Clang) on the primitives removes every call and restores the tight scalar `vaddsd` loop the integer path already had. It is **~23 % faster on double-sum point updates** and ~7 % on XOR point updates — the gain concentrated, as everywhere, in the work-bound point engine where the call sat in the hot loop, and fading to noise on the memory-bound range engine. Integers are unaffected (already inlined); the wrapper just makes the choice uniform across value types.

A subtlety worth recording: floating-point addition is **not associative**, and the segment tree merges in a different order than the naive left-to-right ground truth, so a bit-exact correctness check on rounded values would fail for no fault of the engine. The `SumDouble` benchmark sidesteps it by drawing integer-valued doubles bounded so every partial sum stays below 2⁵³ — exact under any order, while the instruction stream remains genuine double-precision FP. `MaxDouble` uses arbitrary fractional values (max never rounds). All three new types pass the runner's per-query ground-truth check.

### Borrowing the wide tree's trick: a vectorized-update concept

The [external references](#external-reference-implementations-sumint32-point) section left one concrete thing to learn from the Algorithmica wide tree. It propagates a point update with a single branchless AVX2 masked add per level, whereas the generic engine did the same suffix fold as a scalar loop — so its `int32` point-sum update was the heaviest of the group (~49 ns/op average, with Fenwick at ~32 and the wide tree at ~26). And the only real difference between the two on that update is exactly that one step: both are B = 16-ary running-prefix trees, both answer a prefix query with one read per level. A single self-contained technique like that is precisely the kind of thing a generic engine can adopt as an *operation hook* without ceasing to be generic — so we borrowed it.

The hook is `ApplyToMany(block, count, from, delta)`, detected by the `HasApplyToMany` concept — named for *what* it does (merge one value into many accumulators in place), deliberately not for *how* the engine happens to use it, so an operation author need not know anything about the tree's block layout. Its contract is exactly what the point engine's inner loop does — `for (i = from; i < count; ++i) block[i] = Merge(block[i], delta)`, with `block[0 .. from)` readable but left untouched — and the operation is free to implement it however it likes. `TFatPointUpdateEngine::SetPoint` calls it when present and falls back to the scalar loop otherwise, so operations that do not provide one are unaffected. `TSumOperation` provides an AVX2 implementation that is precisely the wide tree's trick: broadcast the delta across the lanes, mask it against a per-lane `lane ≥ from` predicate so the prefix stays untouched, and add the whole fixed-size block in two `vpaddd`s — no inner loop, no variable trip count, no remainder. The earlier finding that vectorizing the *scalar* suffix loop made it slower does not apply here, and for the same reason it failed there: that loop had a runtime trip count and paid head/tail overhead, whereas the masked full-block form is straight-line.

With the borrowed trick the update is no longer the bottleneck. On the mixed `int32` point-sum workload, average mode, interleaved:

| Engine | ns/op |
| --- | ---: |
| generic, scalar suffix loop (before) | 49 |
| Fenwick | 32 |
| **generic + `ApplyToMany` (after)** | **28** |
| Algorithmica wide tree | 26 |

The generic engine's update lands in the same neighbourhood as the two references. The TSC split confirms the mechanism is exactly the intended one: the median *modification* falls from ~108 to ~49 cycles (the vectorized suffix), while the *query* is unchanged (~76 cycles) — the hook touches only the update, which is all that ever separated the two structures on this operation. What remains between it and the wide tree is the query side (the wide tree's fixed-unrolled, cache-line-aligned per-level reads), a separate concern a second hook could address.

The same hook carries to the other value types of this session — `TSumOperation`'s `double` path (a masked `vaddpd`) and `TBitwiseXorOperation`'s `int32` path (a masked `vpxor`; `x ^ 0 == x` makes the identical lane mask correct) — for **~29 % and ~40 %** on their respective point updates. One operation-level method, written once per operation that wants it, gives the universal engine the same vectorized update on each of them, while every operation that does not want it keeps the portable scalar fallback. (One implementation detail the disassembly forced: the update walk now stops at `currentNode > ROOT_NODE_ID` rather than `>=`, because the masked full-block form reads the whole sibling block, and the root's block index would underflow — the root has no siblings, its suffix is empty, and its value is never read by a query, so the iteration was a no-op in the scalar form anyway.)

The residual ~2 ns to the wide tree is the query, and two attempts to close it came up empty. Forcing the prefix walk into a compile-time-unrolled, independent-address form (the wide tree's `for layer < H` shape) did not move it. And 64-byte-aligning the node array — so the masked vector block load/store hits a single cache line instead of straddling two, the one place the wide tree's `alignas(64)` could still matter — measured as pure noise (median of five: 28 vs 29 ns), because here a split load over L1-resident data is essentially free. The update gap was the whole story; the remainder is below the measurement floor.

**What the genericity buys.** The interesting part of borrowing the trick is not the sum number on its own; it is that the *same* hook serves other operations too, from one structure. To make that concrete rather than asserted, `external/algorithmica.h` now also carries a **hand-written XOR wide tree** (`TAlgorithmicaWideSegmentXorEngine`), written for this illustration. The upstream design is, very reasonably, specialized to addition, so adapting it to XOR meant re-deriving and re-writing the whole structure: a second ~110-line class with `^` for `+`, `old ^ new` for the delta, and a masked `vpxor` for the propagation (validated against a brute-force XOR reference in the test suite). The generic engine needed *nothing new* for the same problem — just the one `ApplyToMany` method already on `TBitwiseXorOperation`:

| Operation | generic engine | hand-written wide tree | ratio |
| --- | ---: | ---: | ---: |
| int32 **sum**, point | 29 ns | 28 ns | ~1.04× |
| int32 **XOR**, point | 29 ns | 25 ns | ~1.16× |

So per operation the generic engine runs within a small constant factor of a structure written specifically for it, while a specialized structure has to be re-derived for each new problem (a third time for the next group, a fourth for `double`, and not at all for a non-invertible operation like `max`, whose prefix trick does not exist). That is the trade the project is built around — a small, bounded constant-factor cost in exchange for covering an open-ended family of problems with one tested engine — and it is only this clear because there was a specialized reference to measure it against.

## A second round: specializing the descent

A later optimization pass, run under the same discipline as the [first one](#a-round-of-empirical-optimization) — every change measured interleaved A/B against the previous accepted state, kept only when the measurement defends it — produced one structural change to the range engine, two operation-level changes, and a couple of rejections that are worth recording alongside the wins. The section closes with two measurement artifacts this round surfaced, both of which changed the protocol itself.

### One-sided fringe walks in the range engine

The generic recursive descent treats every node identically: check full coverage against *both* range bounds, and recurse into whichever children intersect the range. But the shape of a range operation is not uniform. The descent from the root first follows a *spine* — the range strictly interior to one child per level — until it reaches the highest node the range straddles; from there it splits into a *left fringe*, which only ever assigns or reads a suffix `[rangeStart, nodeEnd)` of each node it visits, and a *right fringe*, which only ever sees prefixes. The generic code re-derives this from both bounds at every level, and — worse — pays a full recursive call for the fully covered child hanging off each fringe level, only for that call to rediscover at its entry check what the parent already knew.

`ModifyNodes`/`QueryNodes` are now only the spine; at the split node they hand the fringes to one-sided specializations (`ModifySuffix`/`ModifyPrefix`, `QuerySuffix`/`QueryPrefix`). Each fringe level tests a single bound, the fully covered sibling is handled in place — two straight-line writes (lazy tag plus recomputed subtree value) on the modify path, a single adjacent array read on the query path — and the parent's pending lazy tag is no longer pushed into a child that the current assignment overwrites wholesale anyway. The transformation is purely structural: the same nodes are touched, in the same left-to-right merge order the non-commutative operations require.

Interleaved A/B on the main range datasets (average mode, median of five):

| Task, range flavor | before | after | speedup |
| --- | ---: | ---: | ---: |
| sum | 247 ns | 217 ns | **1.14×** |
| max | 303 ns | 238 ns | **1.27×** |
| max-subarray | 507 ns | 405 ns | **1.25×** |
| order statistic (K=16) | 824 ns | 821 ns | 1.00× |

The pattern repeats the first round's lesson: the win is largest where per-node work is small (sum, max — the saved calls and checks are a real fraction of the budget) and fades to nothing where a sixteen-way state merge dwarfs the traversal (the order statistic). The per-operation (fenced) mode shows the same direction, sum ~1.20×.

### A branchless top-K merge

The order-statistic cells are dominated by one loop: merging two descending, distinct top-K sequences. The straightforward three-way if-chain (take left / take right / tie — drop the duplicate) branches on a comparison whose outcome is data-random, so it mispredicts on essentially every iteration, and at K=16 that is sixteen flushes per merge. The merge is now branch-free: each front is read through an "exhausted → `INT32_MIN`" guard, the winner is a `max`, and both cursors advance by comparison results — `cmov`/`setcc`, nothing to predict. A tie takes the value once and advances both cursors, which is exactly the duplicate-dropping the branchy form did; a *real* `INT32_MIN` at a front still works, because the loop condition guarantees at least one cursor is in bounds and an exhausted side's cursor may harmlessly step past its end while the guard keeps reporting the sentinel.

Measured on the order-statistic datasets: **1.17×** on the point flavor (whose queries, per the [workload analysis](#order-statistic-the-gap-is-the-workload-not-the-engine), merge real top-K states down the whole path) and **1.06×** on the range flavor (where most queries are answered from a lazy tag before any merging starts).

### Squeezing the max-subarray operation

Two smaller changes to `TMaxSumStateOperation`, accepted as one unit. Its `FromMany` — the leaf-block summarizer — was a call into the naive engine's two-pass reference (a forward pass, then a *backward* pass for the maximum suffix sum). One forward pass suffices: Kadane's "best subarray ending here", evaluated at the last element, *is* the best non-empty suffix sum. The naive engine deliberately keeps the two-pass form — the ground truth should not share code shapes with the path it validates. And `FromRepeated`, called on every level of the descent when a node is assigned wholesale, branched on the sign of the incoming value — a data-random branch; it is now a ternary that compiles to a `cmov`. Together: **1.12×** on max-subarray point (where every update rebuilds a leaf block through `FromMany`), neutral on the range flavor.

### Rejected by measurement

Consistent with the first round's experience, the rejected hypotheses say as much about the engine as the accepted ones:

- **Dropping the top-K state's zero-initialization.** `TTopKState` value-initializes its sixteen-slot array, which costs two 32-byte stores on every `Merge` temporary; every reader honors `Size`, so the zeroing is semantically dead. Measured: 0.98× on a clean tie-break — the stores were free all along (store-forwarded, off the critical path), and the change traded them for indeterminate-value semantics. Rejected.
- **Unrolling the one-sided query recursion into a descent loop.** The fringe recursion is linear, so it converts mechanically into a loop with a running accumulator. A decisive 15-pair run put the effect at 1.01–1.02× (geomean) for the register-sized states — GCC was already compiling the recursion that well — and a measured **loss** (~0.89×) for the 68-byte top-K state, where the accumulator forces one extra state copy per covered sibling that the recursive build-on-return form never materializes. Rejected in full; the recursive form stays.
- **`FromMany` for max/min.** Rejected without a run: the disassembly shows the generic per-element `Merge` fold already auto-vectorizes into the same `vpmaxsd` reduction a bulk method would hand-write (the release binary carries 2,689 of them). This is the [SIMD discussion](#simd-discussion)'s conclusion arriving from a third direction.

### Two measurement artifacts worth recording

Both of this round's near-misses came from the measurement, not the code, and both now shape the protocol:

- **A freshly linked binary measures slow.** The first two or three runs after a relink came in up to 60% slower than steady state — cold page-cache on the WSL2 DrvFS filesystem, paid by exactly one side of the A/B pair (the side that was just rebuilt). Since the cold side is always the *candidate*, the bias is at least conservative — it can only hide a win, not fabricate one — but it very nearly caused a false rejection. Every A/B batch now starts with a discarded warmup run per binary per dataset.
- **Median-of-five cannot resolve ten percent on this machine.** Two runs of *identical* code, hours apart, produced per-task medians 40% apart in both directions. For any effect under ~10% the decision statistic is now the geometric mean of *within-pair* ratios across at least twelve interleaved pairs — each pair is measured close enough in time that the thermal drift cancels inside it — with the pairwise win count as a sanity check, and the in-pair order alternated (base-first on odd pairs, candidate-first on even) so that even the drift *within* a pair has no preferred side. The first round's interleaving advice was necessary but, for effects this small on hardware this noisy, not sufficient.

### Net effect of the round

The accepted set — the one-sided fringe walks, the branchless top-K merge, and the max-subarray operation changes — measured as one final interleaved comparison against the pre-round binary, on all eight main datasets (average mode; geometric mean of within-pair ratios, the two noisiest cells re-run at twelve alternating pairs):

| Task | point | range |
| --- | ---: | ---: |
| Sum | 1.01× | **~1.18×** |
| Max | 1.02× | **1.48×** |
| Max-subarray sum | ~1.05× | **1.33×** |
| Order statistic (K=16) | **1.15×** | **1.23×** |

The point flavors of sum and max are untouched by construction (they run the fat/point engines, whose code this round did not change) and duly measure as noise — a useful negative control for the protocol itself. Every range cell improves, the cheap operations the most, and the order-statistic point cell gets its branchless merge. The pre-round tables above ([per-operation](#per-operation-latency-barrier-separated) and [throughput](#aggregate-throughput-no-barriers)) still show the older binary's numbers; re-measuring them wholesale on quiet hardware is left for the next full benchmarking session.

## A third round: a wide range-assignment engine

The two rounds above tuned the binary engine. This one asked how far a different shape goes: the profile of the range datasets shows the binary engine mispredicting about **16 branches per operation** (166 million over the 10 million operations of a range dataset), and at a length of 10^6 the whole structure fits in the 8 MB L3 -- so the range engine is bound by its branches at least as much as by memory. Everything below is measured on the same laptop, datasets and length as the tables above; "average" and "per-operation" are the runner's two [measurement modes](#benchmark-results).

### Measuring when the machine is in use

The interleaving protocol of the [first round](#measuring-on-a-thermally-noisy-laptop) turned out not to be enough on a laptop in everyday use. Work on the Windows host is invisible from inside WSL -- its load figures stay near zero -- and it can slow down several rounds in a row by a factor of three or four, enough to move a median and to make unchanged code look 40% faster or slower. Three things were added, in `benchmarks/compare_builds.py`, which runs every benchmark of this README over several builds:

- **A busy-machine probe** (`benchmarks/quiet_probe.cpp`): a fixed mix of arithmetic and random memory accesses, run before and after every measurement. If it comes out more than 15% slower than the best of the recent probes, the run waits for the machine to calm down, and a measurement that ended busy is repeated. Only recent probes count, so a lasting slowdown (a thermal state, a power mode) becomes the new normal instead of stopping the run.
- **The median of the fastest half** of each cell instead of the median of all: other work can only add time.
- **Hardware counters** (`benchmarks/perf_stat.cpp`, a small `perf_event_open` wrapper, since WSL ships no `perf`): the instruction count of identical code differs by 0.00% between runs, and cycles and branch misses by well under 1%, against several percent for wall-clock time.

Noise thresholds come from comparing a build with itself and with the same code shifted in memory (a padding function prepended to every translation unit), since code layout alone moves these numbers by a few percent. With the probe, the median noise of `avg_ns_per_query` fell from 5.5% to 2.5%, of cycles to 0.7%. A change is kept when it improves some cell beyond that threshold and loses in none -- with one deliberate exception, [described below](#the-same-hook-in-the-point-engines-a-trade-off).

### The wide engine

`TWideRangeUpdateEngine` (`include/engine/wide/engine.h`) is a 16-ary range-assignment engine that asks nothing of the operation beyond `FromRepeated` and `Merge`. Leaves hold 16 raw values; a node of level h holds 16 slots, each with the result of a child spanning 16^h elements and, if that whole child was assigned a value it has not seen yet, that value as a lazy tag (one bitmask per node says which slots are tagged). The tree spans a multiple of 16^h elements; a slot reaching past the length holds a valid but meaningless result built from `TValue()`, which is never read -- no operation inside the length can fully cover such a slot -- so no neutral element is needed.

An assignment descends while the range stays inside one slot, then walks its **two boundary paths side by side**, level by level, and refolds them side by side: two independent dependency chains instead of one twice as long. Covered slots and the leaf range are written with fixed-shape selects that compile to blends rather than loops. A query never pushes tags (a tagged boundary slot answers with `FromRepeated`) and is **unrolled by level**: a template parameter makes every span a constant and gives every level its own branches, which the predictor then learns separately.

Against the binary engine at its *best* configuration, not just the defaults of the tables above (leaves of 16 to 256 elements, with and without the sentinel), the wide engine wins on the cheap operations. The wide figures include the hooks [below](#hooks-for-the-wide-engine); the two columns come from separate interleaved runs.

| Range dataset, average mode | Wide | Best binary | Difference |
| --- | ---: | ---: | ---: |
| Sum | 100 ns | 155 ns (leaf 64, sentinel) | -36% |
| Max | 102 ns | 156 ns (leaf 64) | -35% |
| Max-subarray (with `MergeBlock`) | 164 ns | 207 ns (leaf 64) | -21% |
| Order statistic (K=16) | 918 ns | 648 ns (leaf 16, the only one measured) | +42% |

The branch misses tell why: about 8 per operation against 16. Memory is 9% below the binary engine's default configuration and within 4% of its leanest one. The order statistic shows the limit: a 16-slot refold of a 68-byte top-K state is far more work than the two-child merges of a binary node, and nothing in the branch savings makes up for it.

### Hooks for the wide engine

Two optional hooks ([MergeBlock](#mergeblock-hasmergeblock), [FoldInBlock](#foldinblock-hasfoldinblock)) target the two kinds of block work:

- `MergeBlock` on the maximum subarray turned the wide engine from 15% *slower* than the binary default into 21% faster than the best binary configuration, with 22% fewer instructions and 37% fewer branch misses than the latter.
- `FoldInBlock` on the sum and the maximum, against the wide engine without it: the median query fell by 25-27% (68 -> 51 TSC cycles), branch misses by 12%, the average by 6-7%; assignments stayed within noise.

### Which engine by default

`include/engine/default.h` provides `TDefaultRangeUpdateEngine` and `CreateRangeUpdateEngine()`, which pick an engine from the types alone. Whether the wide shape pays off depends on the cost of a merge, which the types cannot tell -- the size of the result is only a rough proxy:

| Result | Size | Wide / best binary |
| --- | ---: | ---: |
| Max, `int32` | 4 B | 0.67 |
| Max, `int64` | 8 B | 0.68 |
| Sum, `int64` | 8 B | 0.62 |
| Sum and max of the same range (cheap merge) | 16 B | 0.82 |
| Maximum subarray (expensive merge, no hook) | 16 B | 1.35 |
| Four cheap fields | 32 B | 0.95 |

At 16 bytes the outcome already depends on the operation, so the default is the wide engine for results of up to 8 bytes, or whenever the operation provides `MergeBlock` -- which says that the fold of a block was made cheap on purpose. Everything else stays binary. A task that knows better simply names the engine it wants.

### The same hook in the point engines: a trade-off

The point engines fold a partial leaf with the same variable-length loop, and `FoldInBlock` applies there too (a partial leaf always lies inside one aligned 16-element block). On streams of operations it is the largest gain of the round: the fat engine's bare prefix query goes from 14.9 to 8.2 ns and its range query from 26.6 to 14.7 ns, the fence experiment's throughput from 13.7 to 7.2 ns, and on the runner's point datasets the average falls by 18% (sum) and 11% (max), with branch misses down 55% and 39%. Updates do not change. This is what the [leaf-prefix experiment](#where-the-generic-engines-query-time-goes-the-fat-leafs-brute-force-scan) promised, without its price in update cost.

It is not free in isolation, though. With fences around every operation the prefix query gets *slower* -- 41 -> 47 ns in the branching benchmark, 42 -> 46 ns in the fence experiment -- because a single short loop that the predictor happens to guess right is quicker than computing a mask. Fenced range queries still improve (by 5-9%). The hook is kept for the throughput it brings, at that cost.

### Building from an array

`InitFrom(values)` builds every engine in O(N) from an array, bottom-up, instead of `Init` followed by one modification per element. A new benchmark (`benchmarks/init_microbench.cpp`) compares the two for each operation and engine of the datasets, and checks the two structures against each other on random queries: `InitFrom` is 14-96 times faster. The point engines also write their value array once rather than zero-filling it first, which made their build 13-21% faster still.

### Rejected in this round

- **Prefetching the paths of upcoming operations.** Unlike the [descent-time prefetch](#what-did-not-work) rejected before, this one ran ahead across operations: the runner read eight operations ahead and asked the engine to start loading the lowest levels of their paths. With the whole structure in L3 there was nothing for it to hide; cycles grew by 0.6-8% and instructions by 3-8%.
- **Hand-tuned binary-engine descents** were put aside rather than rejected: the wide engine covers the cheap operations, where they would have mattered most.

## Library Checker: Range Set Range Composite

`external-benchmarks/library-checker/range_set_range_composite/` holds a solution of [Range Set Range Composite](https://judge.yosupo.jp/problem/range_set_range_composite): assign a linear function `x -> c x + d` to a range, and evaluate the composition of the functions over a range at a point, modulo 998244353, for up to 5 * 10^5 elements and operations. It is a self-contained file built on the same ideas as the wide engine: a 16-ary tree whose nodes are written with masks; folds of a block in AVX2 lanes, with Montgomery arithmetic kept in [0, 2p) and one product per composition; per-assignment records of the powers f, f^2, f^4, ... built 16 at a time in SIMD lanes; and the paths of upcoming operations prefetched. `stress.sh` checks it against a brute force and an independent lazy segment tree.

As of November 11, 2026, it is the fastest accepted solution of the problem on Library Checker: 122 ms on its slowest test ([submission 414997](https://judge.yosupo.jp/submission/414997)).

## Ensuring code quality

The project is done by a single person, without possibility to establish code review process. Essentially, it highlights the need for establishing good practices, among which are:
- An appropriate set of sanitizers is applied, when tests are run. We sanitize addresses, undefined behavior and memory leaks. The thread sanitizer is not applied here: the code is single-threaded and targets squeezing nanoseconds.
- The testing is done as a stress test. There is a simplified test suite though, where the reduced versions of the tests are present. They serve as the first line of defense: allowing to have something very simple and catching the most common bugs.

The stress tests test all instances of the task: the range modification case and the single point modification case. All types of operations are added: commutative and non-commutative, those with the complex state, those with the default element not being neutral.

Once the instance of the task is chosen, the following is done:
- The stress test with `10_000` random modifications is done for all structure sizes from `1` to `16` inclusively. This choice ensures that the corner cases are tested, and no errors take place on small segments.
- Once the smaller lengths are tested, a stress test for the structure size equal to `1_000` is run.
- All combinations of layouts and, where applicable, leaf sizes or branching factors are tested with the methodology above.

## Building and Dependencies

The whole thing is supposed to be buildable with CMake. Conan v2.22.2 was used for the package management. Use your best judgement, if you're familiar with the package manager. Otherwise, here are the key points you need to know in order to make everything running.

First, you will need to configure a Conan build profile. You can do that with `conan profile detect`. If nonexistent before, the profile will be created. You can view it with `conan profile show`. You may equally want to edit certain settings. For instance, you'll need C++23 (specify that in `compiler.cppstd`). You may also want to change the type of compiler.

Note that with the default configuration, one Conan profile means one build mode: either debug or release, but not both. Due to that, if you want to develop the engine or run the tests properly, you will need put in place, another profile, for debugging purposes. You can do that by first, duplicating the existing one with `conan profile detect --name=debug`, and then editing it - in particular you'll need to set `compiler.runtime_type=Debug`.

Once you're done with profiles, you can first checkout the dependencies from the C++ package repository. You can do that with `conan install . --output-folder=build --build=missing --profile=debug` (note that you'll need to omit this `profile=...` part for release). Conan will checkout only one version of a package, either debug or release. Then, prepare CMake with `cmake -B build -DCMAKE_TOOLCHAIN_FILE="build/generators/conan_toolchain.cmake"`. And then, you're ready to build with `cmake --build build` for debug or `cmake --build build --config Release` for release. Make sure that you align your Conan profile with the mode that will be used for the build. Of course, once the preparation is done, once you edit the code, you can run only that last `cmake --build build` command.
