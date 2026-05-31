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
