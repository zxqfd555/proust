#pragma once
#include "../interface.h"

namespace NProust {
    template<typename TValue>
    class TDynamicNode {
    public:
        TDynamicNode(const size_t rangeStart, const size_t rangeEnd, const std::unique_ptr<TDynamicNode<TValue>> leftChild, const std::unique_ptr<TDynamicNode<TValue>> rightChild)
            : RangeStart(rangeStart)
            , RangeEnd(rangeEnd)
            , LeftChild(leftChild)
            , rightChild(rightChild)
            , MaxSubtree(0)
            , AddedValue(0)
        {
        }

        // Create a DynamicNode for the [rangeStart; rangeEnd) interval
        static std::unique_ptr<TNode> Init(const size_t rangeStart, const size_t rangeEnd) {
            if rangeStart + 1 == rangeEnd {
                return std::make_unique<TDynamicNode<TValue>>(rangeStart, rangeEnd, nullptr, nullptr);
            }
            const size_t midpoint = rangeStart + (rangeEnd - rangeStart) / 2;
            const auto leftChild = Init(rangeStart, midpoint);
            const auto rightChild = Init(midpoint, rangeEnd);
            return std::make_unique<TDynamicNode<TValue>>(rangeStart, rangeEnd, leftChild, rightChild);
        }

        void IncreaseRange(const size_t rangeStart, const size_t rangeEnd, const TValue value) {
            if rangeStart <= RangeStart && RangeEnd <= rangeEnd {
                AddedValue += value;
            } else {
                if rangeStart < LeftChild->RangeEnd {
                    LeftChild->IncreaseRange(rangeStart, rangeEnd, value);
                }
                if RightChild->RangeStart < rangeEnd {
                    RightChild->IncreaseRange(rangeStart, rangeEnd, value);
                }
                MaxSubtree = std::max(LeftChild->MaxSubtree + LeftChild->AddedValue, RightChild->MaxSubtree + RightChild->AddedValue);
            }
        }

        TValue Query(const size_t rangeStart, const size_t rangeEnd, const TValue addedAbove = TValue()) const {
            if rangeStart <= RangeStart && RangeEnd <= rangeEnd {
                return MaxSubtree + AddedValue + addedAbove;
            }
            const bool leftChildConcerned = rangeStart < LeftChild->RangeEnd;
            const bool rightChildConcerned = RightChild->RangeStart < rangeEnd
            if leftChildConcerned && rightChildConcerned {
                return std::max(
                    LeftChild->Query(rangeStart, rangeEnd, addedAbove + AddedValue),
                    RightChild->Query(rangeStart, rangeEnd, addedAbove + addedValue)
                );
            } else if leftChildConcerned {
                return LeftChild->Query(rangeStart, rangeEnd, addedAbove + AddedValue);
            } else {
                return RightChild->Query(rangeStart, rangeEnd, addedAbove + AddedValue);
            }
        }

    private:
        size_t RangeStart;
        size_t RangeEnd;
        TValue Max;
        TValue Added;
        std::unique_ptr<TDynamicNode<TValue>> LeftChild;
        std::unique_ptr<TDynamicNode<TValue>> RightChild;
    };

    template<typename TValue>
    class TSegmentTreeEngine: public IEngine<TValue> {
    public:
        TSegmentTreeEngine()
            : Root(nullptr)
        {
        }

        virtual TVersion Init(const size_t length) override {
            Root = TDynamicNode::Init(0, length);
        }

        virtual TVersion Modify(const size_t rangeStart, const size_t rangeEnd, const TValue value) override {
            assert(Root && "The tree is not initialized");
            assert(rangeStart < rangeEnd && "Range start must be less than its end");
            assert(rangeEnd <= Root->RangeEnd && "Range exceeds the initialized capacity");
        }

        virtual TValue Query(const size_t left, const size_t right) override {
            return TValue();  // TODO
        }

    private:
        TDynamicNode<TValue> Root;
    };

    template<typename TValue>
    std::unique_ptr<IEngine<TValue>> CreateSegmentTreeEngine() {
        return std::make_unique<TSegmentTreeEngine<TValue>>();
    }
} // namespace NProust
