# proust

This repo is a deep-dive on building a robust universal range query engine in C++. The next sections contain the details on the design decisions made and efficient implementation.

In brief, the main lines are:
- We use a segment tree as an efficient structure with `O(log N)` time per both modification and query. We suppose that the reader is familiar with this structure and will only describe the optimized and cache-efficient variants.
- In practice, the `O(log N)` time doesn't mean much: two codes with the same asymptotic may differ up to an order of magnitude in terms of both, throughput and the tail latency. Most of the effort is concentrated on obtaining practical results and reliable benchmarking.

## Table of Contents

- [The Task](#the-task)
- [Examples](#examples)
- [Interfaces](#interfaces)
- [Optional operation extensions](#optional-operation-extensions)
  - [Undo (HasUndo)](#undo-hasundo)
  - [FromMany (HasFromMany)](#frommany-hasfrommany)
    - [SIMD discussion](#simd-discussion)
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
- [External baselines (SumInt32, point)](#external-baselines-sumint32-point)
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

The reason there are exactly two is that they address the two different places where the generic path is wasteful: the *query strategy* on point-update engines, and the *leaf work* at the bottom of any engine. They are independent, and a given operation may expose neither, either, or both (the integer sum, for instance, exposes both).

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

In practice the case for doing it by hand turned out to be hard to justify, because the blocks `FromMany` actually sees are small. A fat leaf spans `LeafSize` elements and a partial leaf fewer, and the whole purpose of the surrounding machinery — cached subtree values, lazy tags, prefix walks — is to *avoid* scanning long runs in the first place. The vectorizable part is therefore a short tail, dominated by call and reduction overhead rather than by steady-state throughput.

We tested it regardless. We added hand-written AVX2 variants of both the sum and the maximum/minimum operations and benchmarked them against the scalar versions, pushing the leaves all the way to `LeafSizeLog = 10` — blocks of 1024 elements — to hand the vector code as much contiguous work as the structure allows. The outcome was no measurable gain at any leaf size, in either the range or the point engine; at the largest leaves the explicit AVX2 was, if anything, marginally slower.

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

The conclusion is that we did not implement SIMD for any of the provided operations ourselves. The compiler vectorizes the simple reductions for us, and the structure never feeds them runs long enough for a hand-tuned kernel to earn its keep. The `HasFromMany` interface nevertheless leaves the door fully open: an operation whose block summary is genuinely expensive, or a deployment that deliberately runs very fat leaves, is free to supply a vectorized `FromMany`, and the engine will pick it up with no other change.

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

**Two measurement modes.** The per-operation `RDTSC` instrumentation is not free, but the cost is subtler than "the fences are slow", and getting it right is what motivates the second mode. The `LFENCE`/`RDTSCP` pair around each operation has two distinct effects. The first is a *fixed* cost — the instructions themselves — and it is small: timed in isolation (fences around a no-op) it comes to about **24 ns** on this machine, tens of nanoseconds, and the same for every engine. That fixed cost alone cannot explain a per-engine spread of hundreds of nanoseconds, and it doesn't. The second effect is the one that matters: the fences are *serializing*, so each operation must fully retire before the next begins. That does not add an overhead so much as it changes *what is measured*. A bare loop measures **throughput** — the CPU overlapping one operation's cache misses with the next's; a fenced loop measures **latency** — one operation start-to-finish, alone. For an operation that is mostly independent memory accesses those two numbers diverge enormously, and the gap is a property of the algorithm, not of `RDTSC`.

A small experiment (`benchmarks/fence_experiment.cpp`) makes it concrete — the prefix-sum query timed in a bare loop versus a fenced loop:

| engine | throughput (bare) | latency (fenced) |
| --- | ---: | ---: |
| robust (fat) | ~17 ns/op | ~56 ns/op |
| fenwick | ~16 ns/op | ~60 ns/op |
| wide tree | ~6 ns/op | ~60 ns/op |

Fenced, all three converge to ~60 ns — essentially one serialized cache miss plus the ~24 ns fence floor — so per-operation timing makes them look *alike*. Unfenced, they spread apart: the wide tree's query is little more than a handful of independent loads, so the out-of-order engine keeps roughly ten of them in flight and throughput falls to ~6 ns, while the engines whose queries carry data-dependent work and branches overlap less and stay near ~16 ns. Per-operation timing therefore flatters the dependency-bound engines and hides the wide tree's real advantage; only a throughput measurement exposes it. The runner accordingly offers both, chosen with `--measurement-mode`:

- `per-operation` (the default): every operation is bracketed by the fences, yielding the per-operation TSC histograms and the modify/query split. It reports *latency*. Use it when operations are individually expensive — so neither the fixed cost nor the lost overlap matters much — or when the per-operation distribution is what you want.
- `average`: the instrumentation is dropped entirely; whole batches are timed with the wall clock and divided by the operation count. It reports *throughput*, operations free to overlap — at the price of the per-operation breakdown, which is not collected (its TSC columns are reported as zero).

The choice is not academic. Replaying the mixed int32 point-sum dataset, the wide tree measures slower than the generic engine in `per-operation` mode (≈230 vs ≈120 ns/op — its memory-heavy updates have the longest serialized latency) yet clearly *faster* in `average` mode (≈40 vs ≈70 ns/op, with Fenwick at ≈47) — the same code, answering two different questions. For the genuinely expensive range engine, whose operations cost hundreds of nanoseconds and overlap little, the two modes agree to within noise. And this is exactly why the average mode matters for the comparison below: the state-of-the-art prefix-sum structures spend almost all their time on overlappable memory accesses, so latency-style per-operation timing is the wrong instrument for them, and only throughput gives them a fair hearing. Rule of thumb: `per-operation` when the operation is expensive or you need percentiles; `average` when it is cheap and heavily pipelined.

**Benchmarking a universal engine against specialized state of the art.** The engine here is deliberately universal: the value type and the result type may differ, and any associative operation with the right hooks plugs in. That generality is the point of the project, but it raises an obvious question — how much does it cost against a structure built for exactly one problem? To answer it we take the one partial case that has famous, hand-tuned competitors — point modifications plus prefix/range sums over an `int32` array — and benchmark the generic engine against two external baselines (see [External baselines](#external-baselines-sumint32-point)): a handwritten Fenwick tree, the textbook answer, and the **wide segment tree** from Sergey Slotin's Algorithmica/HPC, the current state of the art for prefix sums. The wide tree is fast because it is a wide (`B = 16`) tree laid out as a flat array of cache-line-sized nodes holding running prefix sums, and it propagates a point update with a single branchless **AVX2** masked add per level — SIMD doing the propagation that a scalar tree performs one node at a time. It is entirely expected, and entirely fine, that these specialized structures win: they solve a narrower problem and should. The goal is not to beat them but to put the generic engine on the same axis and confirm it stays *comparable* — within a small factor — rather than landing in a different league. (It does, and on some operations it even leads; the numbers are in the linked section.)

One setup detail earns its own mention, because it turned out to matter. The engines were originally driven through the `IPointUpdateEngine` / `IRangeUpdateEngine` virtual interface — one indirect call per operation. Holding each engine by its concrete type instead, so the hot-path calls devirtualize and inline (the implementations are marked `final` to guarantee it), improved the generic engine's own query latency by roughly 15–30%, and the wide tree's by about 70%. Virtual dispatch, in other words, was not free even for our work-bound engine, and was outright costly for the latency-bound wide tree. That devirtualization is unconditional and was kept; the per-operation-versus-average decision above is the part deliberately left to the caller.

## Benchmark results

The numbers below were produced by `benchmarker.py` (the `execute` and `report` modes), replaying the eight datasets — four tasks, each in a point and a range flavor — seven times per cell against the `robust` engine (`SoA` layout, binary-heap topology, leaf-size-log 4). Each row shows the representative run (the median of the seven by 99.999th-percentile wall-clock latency); the full per-percentile tables, including the TSC-cycle splits and 95% confidence intervals, live in [`benchmark_report.md`](benchmark_report.md). The runner records the processor in every CSV row and prints it on startup, so a result is never separated from the hardware that produced it.

**Processor.** `11th Gen Intel(R) Core(TM) i5-1130G7 @ 1.10 GHz` — a four-core / eight-thread, 15 W Tiger Lake part with a 1.10 GHz base and a ~4.0 GHz single-core turbo, and an 8 MB L3. The benchmarks were run without the `isolcpus` isolation the methodology section recommends, so treat the absolute latencies as indicative rather than definitive, and the *relative* comparisons between tasks and modification flavors as the real takeaway. Latencies above 1,000 ns are reported in microseconds (`us`).

| Task | Modification | Avg latency | Wall p50 | Wall p99 | Wall p99.9 | Peak memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Sum | point | 238 ns | 187 ns | 806 ns | 4.33 us | 13.2 MB |
| Sum | range | 621 ns | 524 ns | 1.75 us | 6.09 us | 13.2 MB |
| Max | point | 169 ns | 155 ns | 408 ns | 801 ns | 19.4 MB |
| Max | range | 617 ns | 495 ns | 1.85 us | 7.80 us | 13.2 MB |
| Max-subarray sum | point | 357 ns | 312 ns | 929 ns | 2.14 us | 43.6 MB |
| Max-subarray sum | range | 556 ns | 504 ns | 1.23 us | 2.61 us | 14.4 MB |
| Order statistic (K=16) | point | 2.55 us | 2.34 us | 4.50 us | 8.97 us | 147.6 MB |
| Order statistic (K=16) | range | 1.23 us | 1.14 us | 2.50 us | 4.87 us | 20.9 MB |

A few things stand out:

- **Point modifications are the cheaper subcase, as predicted — except where state dominates.** For sum, max and max-subarray the point flavor wins on median latency (155–312 ns versus ~500 ns), confirming the smaller constant factor of the `RangeStart = RangeEnd - 1` specialization. The order statistic inverts this: its point flavor is *slower* (2.34 us versus 1.14 us at p50) and far hungrier on memory (147.6 MB versus 20.9 MB). The point-update engine materializes the full sixteen-wide top-K state at every node instead of leaning on the range engine's lazy-tag compression, so it trades the algorithmic saving for a much larger working set and the cache misses that come with it.
- **The TSC split shows where the time goes.** Separating modification from query cycles (the reason for using RDTSC at all) reveals that range assignments are modification-bound: for sum the median assignment costs ~1,460 TSC cycles against only ~230 for a query, because writing a half-open interval has to push lazy tags down `O(log N)` nodes while a query merges a comparable number of already-summarized subtrees. Point modifications, by contrast, are balanced (sum: ~237 modify versus ~266 query cycles).
- **The cost grows with the result type, not just the algorithm.** Sum and max carry a single integer per node and stay in the low hundreds of nanoseconds; max-subarray's four-field state roughly doubles that; the sixteen-element order-statistic state lands in the microseconds. The merge work and the bytes moved per node track the state size directly.
- **Tail latency is environmental.** Across every cell the medians are stable but the p99.9 figures jump into the microseconds. This is the OS jitter the methodology section warns about — interrupts still land on the pinned core without `isolcpus`, and dataset I/O adds its own noise. The wide confidence intervals on the deep percentiles in `benchmark_report.md` say the same thing: the tails describe the machine, not the engine.

## External baselines (SumInt32, point)

A general-purpose engine is only as convincing as the specialized structures it is measured against. For the one partial case that has famous, hand-tuned competitors — point modifications plus range sums over an `int32` array — `include/engine/external/` holds two external reference implementations, selectable in the runner via `--implementation`:

- **`fenwick`** (`external/fenwick.h`) — a textbook Fenwick tree (binary indexed tree), the classic answer to point-update / prefix-sum. A range sum is `prefix(r) - prefix(l)`.
- **`algorithmica`** (`external/algorithmica.h`) — a port of the **wide segment tree**, the fastest structure in Sergey Slotin's [Algorithmica/HPC segment-tree chapter](https://en.algorithmica.org/hpc/data-structures/segment-trees/) (file `segtree/simd.cc` in [sslotin/amh-code](https://github.com/sslotin/amh-code/tree/main/segtree)). It is a B=16-ary tree of cache-line-sized nodes storing running prefix sums; a prefix query is one read per level, and an update is one branchless AVX2 masked add per level. The only adaptation is ownership and a slot count padded up to the next power of B, so the whole-array query `sum(length)` stays in bounds.

Both are fixed to the 10⁶ dataset length, accumulate in `int32` to match the ground truth bit-for-bit, and are validated against the naive engine in `tests/test_external_fenwick.cpp` and `tests/test_external_algorithmica.cpp`. The comparison engine ("best") is what the robust factory selects for sum + point: since `TSumOperation` has `Undo`, that is the `TFatPointUpdateEngine` (branching 4), not the plain segment tree.

**The wide segment tree is the fastest of the three — and measuring that correctly is itself instructive.** The main `runner` reports all three within ~30% of each other (≈100–140 ns/op), with the robust engine nominally ahead. That ranking is an *artifact of measurement*, not reality, for two compounding reasons: the runner used to hold the engine through the `IPointUpdateEngine` **virtual interface** (a per-op indirect call), and — inside the same wall-clock-timed batch — it brackets every operation in `StartRDTSC`/`EndRDTSC`, i.e. `LFENCE` + `RDTSCP` serializing **fences**. The wide tree's query is latency-bound — a few scattered reads, almost no arithmetic — so its speed depends on the CPU overlapping consecutive queries' cache misses (memory-level parallelism), and both of those barriers destroy that overlap.

The first of those two is now fixed: the runner holds each engine by its **concrete** type (`RunDataset` builds it and threads it through a `RunBenchmarkLoop` templated on the concrete engine; the engine classes are marked `final`), so the hot-path calls devirtualize. That alone moves the wide tree's measured query from ~233 to ~186 TSC — real, but small — while the wall-clock average barely budges. The reason is what the [methodology](#benchmarking-methodology) lays out: per-operation timing measures serialized *latency*, not *throughput*. The fixed fence cost is only ~24 ns; the rest is the cross-query overlap the `RDTSCP` forbids by forcing each op to retire before the next issues. Measured that way the wide tree's prefix query is ~60 ns (one serialized cache miss), an order of magnitude above its overlapped throughput of ~5 ns — and at that point it is indistinguishable from Fenwick and the generic engine, which serialize to about the same latency. So the full fix needs the other half too — batch-only timing instead of per-op RDTSC — which is exactly the runner's `average` [measurement mode](#benchmarking-methodology): drop the fences, time whole batches, divide by the count. In that mode the wide tree finally measures ≈40 ns/op and takes its rightful place at the front, ahead of Fenwick (≈47) and the generic engine (≈70). Devirtualizing was the cheap, unconditional win and is always on; the `average` mode is the deliberate second half, kept opt-in because the per-operation TSC split it gives up is what the project's own engines are tuned against.

A tight-loop microbenchmark (`benchmarks/external_microbench.cpp`, no fences, no streaming) shows the true picture. Median ns/op over 5M ops, length 10⁶, identical workload for every contender:

| Engine | update | range `[l,r)` | prefix `[0,r)` |
| --- | ---: | ---: | ---: |
| robust fat (best) | 72 | 38 | 22 |
| fenwick | 36 | 30 | 14 |
| algorithmica — via `IPointUpdateEngine` | 28 | 30 | 29 |
| **algorithmica — devirtualized (concrete type)** | **29** | **10** | **5** |
| raw upstream `simd.cc` (free functions) | 28 | 8 | 4 |

The bottom two rows are the *same engine code*, differing only in whether the call goes through the virtual interface. Devirtualized, the wide tree's prefix query collapses from 29 ns to **5 ns** and its range query from 29 ns to **10 ns** — landing right next to the raw upstream (4 ns / 8 ns), which validates the port — and it becomes the fastest structure by a wide margin: ≈3× Fenwick and ≈4–7× the robust engine on queries, while tying the fast field on updates (where the robust fat engine, at ~70 ns, is the slowest). So the article's claim holds on this very machine; the earlier conclusion that the wide tree "comes last" was a benchmarking error.

The lesson is about the engine's *architecture*, not the wide tree: a polymorphic, one-call-per-operation interface is the right abstraction for the project's own engines (whose per-op cost is large enough to absorb it), but it imposes a hard floor that a state-of-the-art, latency-bound SIMD structure cannot reach. Extracting that structure's speed requires either devirtualizing the hot path or a batched query API that keeps many independent reads in flight — neither of which the current `IPointUpdateEngine` offers. The external baselines thus do more than validate the engine: they mark exactly where its abstraction starts to cost.

Crucially, that cost is *specific to the wide tree*, which is what justifies the interface rather than condemning it. Running the same devirtualized-vs-virtual comparison on the robust fat engine and on Fenwick moves them by ~15% at most (and nothing outside the noise on update or range) — only the wide tree swings 3–6×. The reason is what makes the virtual call expensive in the first place: it is a barrier to issuing the *next* query's loads early, so it only bites operations whose entire budget is cross-call memory-level parallelism. The wide tree's query — about five scattered reads and almost no arithmetic — is exactly that; the robust fat engine's query is work-bound (a tree descent with real per-node merges and dependent loads, tens of nanoseconds of genuine work), so an indirect call is a rounding error against it and there is no idle memory parallelism to reclaim. The polymorphic interface is therefore effectively free for the engines it was designed for, and a ceiling only for a structure that has optimized its per-query work down to nothing.

## Ensuring code quality

The project is done by a single person, without possibility to establish code review process. Essentially, it highlights the need for establishing good practices, among which are:
- An appropriate set of sanitizers is applied, when tests are run. We sanitize addresses, undefined behavior and memory leaks. The thread sanitizer is not applied here: the code is single-threaded and targets squeezing nanoseconds.
- Basic tools, such as clang-tidy are put in place.
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
