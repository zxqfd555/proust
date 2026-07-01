// Low-noise A/B of the max-subarray point engine: the pre-refactor 2N per-element layout
// ("before") vs the production leaf-group layout ("after"), in one binary. No dataset streaming,
// no per-query validation, no RDTSC fences — populate once, then time tight query/update loops.
#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "engine/interface.h"
#include "engine/operations.h"
#include "engine/robust/engine.h"
#include "examples/leetcode_maximum_subarray.h"

using namespace NProust;
using NProustExamples::TMaxSumState;
using NProustExamples::TMaxSumStateOperation;

// ===== BEFORE engine: verbatim pre-refactor TRobustPointUpdateEngine, renamed =====
namespace NBefore {
using namespace NProust;
    template<typename TValue, typename TOperationResult, typename TOperation, size_t LeafSizeLog = 1>
    class TRobustPointBefore final : public IPointUpdateEngine<TValue, TOperationResult, TOperation> {
    private:
        // We use 1-indexing here: although not as nice, it allows us a bit cheaper topology operations
        // as compared to 0-indexing.
        static constexpr size_t ROOT_NODE_ID = 1;
        static constexpr size_t LEAF_SIZE = 1 << LeafSizeLog;
    
    public:
        virtual void Init(const size_t length) override {
            // Overbook capacities up to the closest power of two
            AllocatedLength = std::bit_ceil(length);
            if (AllocatedLength < LEAF_SIZE) {
                AllocatedLength = LEAF_SIZE;
            }

            // Being a power of two, the tree takes 2x nodes from the length
            NodeValues.resize(AllocatedLength << 1);

            // The raw point values are kept alongside the tree so the FromMany fast path
            // can rebuild a leaf's result from the original values (not from merged results).
            PointValues.resize(AllocatedLength);
            std::fill(PointValues.begin(), PointValues.end(), TValue());

            // Initialize the tree with the default values starting from the leafs
            const auto leafValue = TOperation::template FromRepeated<TValue, TOperationResult>(TValue(), 1);
            std::fill(NodeValues.begin(), NodeValues.end(), leafValue);
            for (size_t index = AllocatedLength - 1; index >= 1; --index) {
                NodeValues[index] = TOperation::Merge(NodeValues[(index << 1)], NodeValues[(index << 1) + 1]);
            }
        }

        virtual void SetPoint(const size_t position, const TValue& value) override {
            assert(position < AllocatedLength && "The position is beyond the allocated size");
            size_t nodePosition = AllocatedLength + position;

            // Keep the raw value current before any leaf rebuild: the FromMany path in
            // ComputeRangeBruteForce reads PointValues, so it must already reflect this update.
            PointValues[position] = value;

            const size_t largeLeafStart = position & ~(LEAF_SIZE - 1);

            if constexpr (HasUndo<TOperation, TOperationResult>) {
                const auto oldValue = NodeValues[nodePosition];
                NodeValues[nodePosition] = TOperation::template FromRepeated<TValue, TOperationResult>(value, 1);
                if constexpr (LeafSizeLog > 0) {
                    nodePosition >>= LeafSizeLog;
                    const auto oldValueUndone = TOperation::Undo(NodeValues[nodePosition], oldValue);
                    NodeValues[nodePosition] = TOperation::Merge(oldValueUndone, TOperation::template FromRepeated<TValue, TOperationResult>(value, 1));
                }
            } else {
                NodeValues[nodePosition] = TOperation::template FromRepeated<TValue, TOperationResult>(value, 1);
                if constexpr (LeafSizeLog > 0) {
                    NodeValues[nodePosition >>= LeafSizeLog] = ComputeRangeBruteForce(largeLeafStart, largeLeafStart + LEAF_SIZE);
                }
            }

            while (nodePosition > 1) {
                NodeValues[nodePosition >> 1] = TOperation::Merge(NodeValues[nodePosition & ~1], NodeValues[nodePosition | 1]);
                nodePosition >>= 1;
            }
        }

        virtual TOperationResult ComputeRange(const size_t rangeStart, const size_t rangeEnd) const override {
            assert(rangeStart <= rangeEnd && "Range start can't be bigger than its end");
            assert(rangeStart != rangeEnd && "The operation result for an empty set is undefined");
            assert(rangeEnd <= AllocatedLength && "Range goes beyond the allocated length");

            if constexpr (HasUndo<TOperation, TOperationResult>) {
                const auto fullPrefixResult = ComputePrefixUnaligned(rangeEnd);
                if (rangeStart == 0) [[unlikely]] {
                    return fullPrefixResult;
                }
                const auto irrelevantPrefixResult = ComputePrefixUnaligned(rangeStart);
                return TOperation::Undo(fullPrefixResult, irrelevantPrefixResult);
            } else {
                const auto alignedStart = ((rangeStart + LEAF_SIZE - 1) >> LeafSizeLog) << LeafSizeLog;
                const auto alignedEnd = (rangeEnd >> LeafSizeLog) << LeafSizeLog;
                if (alignedStart >= alignedEnd) [[unlikely]] {
                    return ComputeRangeBruteForce(rangeStart, rangeEnd);
                }

                const auto centralResult = ComputeAlignedRange(alignedStart, alignedEnd);
                if (alignedStart > rangeStart) [[likely]] {
                    const auto leftOffcutResult = ComputeRangeBruteForce(rangeStart, alignedStart);
                    const auto leftCenterResult = TOperation::Merge(leftOffcutResult, centralResult);

                    if (alignedEnd < rangeEnd) [[likely]] {
                        const auto rightOffcutResult = ComputeRangeBruteForce(alignedEnd, rangeEnd);
                        return TOperation::Merge(leftCenterResult, rightOffcutResult);
                    } else {
                        return leftCenterResult;
                    }
                } else {
                    if (alignedEnd < rangeEnd) [[likely]] {
                        const auto rightOffcutResult = ComputeRangeBruteForce(alignedEnd, rangeEnd);
                        return TOperation::Merge(centralResult, rightOffcutResult);
                    } else {
                        return centralResult;
                    }
                }
            }
        }

    private:
        std::vector<TOperationResult> NodeValues;
        std::vector<TValue> PointValues;
        size_t AllocatedLength;

        inline TOperationResult ComputeRangeBruteForce(const size_t rangeStart, const size_t rangeEnd) const {
            assert(rangeStart < rangeEnd);
            if constexpr (HasFromMany<TOperation, TValue, TOperationResult>) {
                return TOperation::template FromMany<TOperationResult>(PointValues.data(), rangeStart, rangeEnd);
            } else {
                auto result = NodeValues[AllocatedLength + rangeStart];
                for (size_t index = AllocatedLength + rangeStart + 1; index < AllocatedLength + rangeEnd; ++index) {
                    result = TOperation::Merge(result, NodeValues[index]);
                }
                return result;
            }
        }

        inline TOperationResult ComputePrefixUnaligned(const size_t prefix) const {
            const auto alignedBorder = (prefix >> LeafSizeLog) << LeafSizeLog;
            if (alignedBorder == 0) [[unlikely]] {
                return ComputeRangeBruteForce(0, prefix);
            }

            const auto mainResult = ComputeRangeBackwards(AllocatedLength, alignedBorder);
            if constexpr (LeafSizeLog == 0) {
                return mainResult;
            }

            if (alignedBorder == prefix) [[unlikely]] {
                return mainResult;
            }
            return TOperation::Merge(mainResult, ComputeRangeBruteForce(alignedBorder, prefix));
        }

        inline TOperationResult ComputeAlignedRange(const size_t rangeStart, const size_t rangeEnd) const {
            // Here we construct the value of the target function by composing the related nodes
            // from the segment tree. The base of this method is recursive (see range modification engine for
            // reference), but since we don't have lazy tags, we can construct the exact set of nodes that
            // composes the answer without recursion, moreover, with the usage of cheap binary operations.
            //
            // As a reward, there's a 4x speedup (even before SIMD and large leaf optimization) for being
            // efficient.
            //
            // No matter the structure of the tree, we use a binary heap layout. We need to take
            // any node starting at `rangeStart` coordinate, and we basically want to take the longest
            // one, not exceeding the query range. At least the node indexed as `AllocatedLength + rangeStart`,
            // an associate leaf, will do for us.
            auto currentNode = AllocatedLength + rangeStart;
            auto queryLength = rangeEnd - rangeStart;

            // Due to heap topology, each node has a contiguous range of power-2 segments starting there,
            // the cardinality of this set is same as the number of zeros at the end of node representation.
            // We refer to it in the code as `maxJump`.
            size_t maxJump = currentNode - (currentNode & (currentNode - 1));
            if (maxJump <= queryLength) {
                auto result = NodeValues[currentNode / maxJump];
                queryLength -= maxJump;
                currentNode += maxJump;
                maxJump = currentNode - (currentNode & (currentNode - 1));

                // While maxJump is smaller than the queried range, the things are simple:
                // we just take the largest node possible, reuse its result, and proceed.
                // The query length decreases, and the leftmost matching node index increases.
                while (maxJump <= queryLength) {
                    result = TOperation::Merge(result, NodeValues[currentNode / maxJump]);
                    queryLength -= maxJump;
                    currentNode += maxJump;
                    maxJump = currentNode - (currentNode & (currentNode - 1));
                }

                if (queryLength > 0) [[likely]] {
                    // At some point, we're in a major node that allows us longer jumps that we need.
                    // It means, we can no longer act in the same way, and we need to dismantle
                    // the remaining length by its' powers of two, and perform the same jumps backwards.
                    // Since `maxJump` is bigger that the whole interval, we can assume that those nodes
                    // will always exist.

                    return TOperation::Merge(result, ComputeRangeBackwards(currentNode, queryLength));
                } else {
                    return result;
                }
            }

            return ComputeRangeBackwards(currentNode, queryLength);
        }

        inline TOperationResult ComputeRangeBackwards(size_t currentNode, size_t queryLength) const {
            // We move to the end of the desired interval and go backwards.
            size_t lastNode = currentNode + queryLength;

            // We pick the length of the last node that will participate in our greedy movement.
            size_t lastStep = queryLength - (queryLength & (queryLength - 1));
            size_t prevNode = lastNode - lastStep;

            // And use the result from that node.
            auto result = NodeValues[prevNode / lastStep];
            lastNode = prevNode;

            // Basically subtract the length of the taken interval, but faster.
            queryLength &= queryLength - 1;

            // Repeat the process when needed.
            while (queryLength > 0) {
                lastStep = queryLength - (queryLength & (queryLength - 1));
                prevNode = lastNode - lastStep;
                result = TOperation::Merge(NodeValues[prevNode / lastStep], result);  // Note the unusual operand order: the operation is non-commutative!
                lastNode = prevNode;
                queryLength &= queryLength - 1;
            }
            return result;
        }
    };
} // namespace NBefore

using Clock = std::chrono::steady_clock;
static double NsPerOp(Clock::time_point a, Clock::time_point b, size_t q){ return std::chrono::duration<double,std::nano>(b-a).count()/q; }

template<typename TEngine>
static std::unique_ptr<TEngine> MakePopulated(size_t N, size_t Q, const std::vector<int>& pos, const std::vector<int>& val){
    auto e=std::make_unique<TEngine>(); e->Init(N);
    for(size_t i=0;i<Q;++i) e->SetPoint(pos[i],val[i]);
    return e;
}
template<typename TEngine>
static double QueryLoop(TEngine& e, size_t Q, const std::vector<int>& ql, const std::vector<int>& qr, int64_t& cs){
    int64_t c=0; auto t0=Clock::now();
    for(size_t i=0;i<Q;++i) c^=e.ComputeRange(ql[i],qr[i]).MaxSubarraySum;
    auto t1=Clock::now(); cs^=c; return NsPerOp(t0,t1,Q);
}
template<typename TEngine>
static double UpdateLoop(TEngine& e, size_t Q, const std::vector<int>& pos, const std::vector<int>& val){
    auto t0=Clock::now(); for(size_t i=0;i<Q;++i) e.SetPoint(pos[i],val[i]); auto t1=Clock::now();
    return NsPerOp(t0,t1,Q);
}

int main(){
    const size_t N=1'000'000, Q=2'000'000;
    std::mt19937 rng(777);
    std::uniform_int_distribution<int> pd(0,(int)N-1), vd(-100,100);
    std::vector<int> pos(Q),val(Q),ql(Q),qr(Q);
    for(size_t i=0;i<Q;++i){ pos[i]=pd(rng); val[i]=vd(rng); int a=pd(rng),b=pd(rng); ql[i]=std::min(a,b); qr[i]=std::max(a,b); if(ql[i]==qr[i])++qr[i]; }
    volatile int t=0; for(size_t i=0;i<Q;++i) t^=pos[i]^val[i]^ql[i]^qr[i]; (void)t;

    using TBefore = NBefore::TRobustPointBefore<int32_t,TMaxSumState,TMaxSumStateOperation,4>;
    using TAfter  = TRobustPointUpdateEngine<int32_t,TMaxSumState,TMaxSumStateOperation,4>;
    auto eb = MakePopulated<TBefore>(N,Q,pos,val);
    auto ea = MakePopulated<TAfter >(N,Q,pos,val);

    size_t mm=0;
    for(size_t i=0;i<200000;++i){ auto rb=eb->ComputeRange(ql[i],qr[i]); auto ra=ea->ComputeRange(ql[i],qr[i]);
        if(rb.MaxSubarraySum!=ra.MaxSubarraySum||rb.MaxPrefixSum!=ra.MaxPrefixSum||rb.MaxSuffixSum!=ra.MaxSuffixSum||rb.FullSum!=ra.FullSum) ++mm; }
    std::printf("max-subarray L4, N=%zu Q=%zu. correctness: %zu mismatches\n", N, Q, mm); if(mm) return 1;

    int64_t cs=0; double qb=1e18,qa=1e18,ub=1e18,ua=1e18;
    // INTERLEAVED best-of: alternate before/after each round so thermal drift cancels.
    for(int r=0;r<9;++r){
        qb=std::min(qb,QueryLoop(*eb,Q,ql,qr,cs));
        qa=std::min(qa,QueryLoop(*ea,Q,ql,qr,cs));
        ub=std::min(ub,UpdateLoop(*eb,Q,pos,val));
        ua=std::min(ua,UpdateLoop(*ea,Q,pos,val));
    }
    std::printf("QUERY : before(2N) %6.1f ns | after(group) %6.1f ns | after/before %.2fx\n", qb, qa, qa/qb);
    std::printf("UPDATE: before(2N) %6.1f ns | after(group) %6.1f ns | after/before %.2fx\n", ub, ua, ua/ub);
    std::printf("(cs %lld)\n",(long long)cs);
    return 0;
}
