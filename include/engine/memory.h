#pragma once

#include <cstddef>
#include <cstdint>
#include <new>

#if defined(__linux__)
    #include <sys/mman.h>
#endif

namespace NProust {
    // An allocator for the large arrays of an engine: from 2 MB up, the memory comes straight from
    // mmap, aligned to 2 MB and marked for transparent huge pages where the kernel allows them
    // (madvise mode). Queries jump around megabytes of nodes at random, and with 4 KB pages a good part
    // of each miss is the page walk; it also faults the array in 2 MB at a time instead of 4 KB at a
    // time. Smaller arrays, and other systems, take the usual operator new.
    template<typename T>
    struct THugePageAllocator {
        using value_type = T;

        static constexpr size_t HUGE_PAGE = size_t(1) << 21;

        THugePageAllocator() noexcept = default;

        template<typename U>
        THugePageAllocator(const THugePageAllocator<U>&) noexcept {
        }

        T* allocate(const size_t count) {
            const size_t bytes = count * sizeof(T);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
            if (bytes >= HUGE_PAGE) {
                // Over-map by a huge page and give back the unaligned ends.
                const size_t mapped = RoundUp(bytes) + HUGE_PAGE;
                void* region = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (region == MAP_FAILED) {
                    throw std::bad_alloc();
                }
                char* start = static_cast<char*>(region);
                char* aligned = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(start) + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1));
                if (aligned > start) {
                    munmap(start, static_cast<size_t>(aligned - start));
                }
                char* end = start + mapped;
                char* alignedEnd = aligned + RoundUp(bytes);
                if (end > alignedEnd) {
                    munmap(alignedEnd, static_cast<size_t>(end - alignedEnd));
                }
                madvise(aligned, RoundUp(bytes), MADV_HUGEPAGE);  // advisory: correct without it
                return reinterpret_cast<T*>(aligned);
            }
#endif
            return static_cast<T*>(::operator new(bytes, std::align_val_t(alignof(T) > 64 ? alignof(T) : 64)));
        }

        void deallocate(T* pointer, const size_t count) noexcept {
            const size_t bytes = count * sizeof(T);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
            if (bytes >= HUGE_PAGE) {
                munmap(pointer, RoundUp(bytes));
                return;
            }
#endif
            ::operator delete(pointer, std::align_val_t(alignof(T) > 64 ? alignof(T) : 64));
        }

        template<typename U>
        bool operator==(const THugePageAllocator<U>&) const noexcept {
            return true;
        }

    private:
        static size_t RoundUp(const size_t bytes) noexcept {
            return (bytes + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1);
        }
    };
} // namespace NProust
