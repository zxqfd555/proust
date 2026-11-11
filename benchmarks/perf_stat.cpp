// perf_stat COMMAND [ARGS...]: runs the command and prints its user-space hardware counters to
// stderr, one "name value" per line. A small stand-in for `perf stat` on systems that do not ship
// the tool (WSL), used by compare_builds.py: the instruction count in particular is nearly free of
// the thermal noise that wall-clock numbers on a laptop carry.
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
    struct TCounter {
        const char* Name;
        uint32_t Type;
        uint64_t Config;
        int Fd = -1;
    };

    int OpenCounter(const TCounter& counter, const pid_t pid) {
        perf_event_attr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.size = sizeof(attr);
        attr.type = counter.Type;
        attr.config = counter.Config;
        attr.disabled = 1;
        attr.enable_on_exec = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        return static_cast<int>(syscall(SYS_perf_event_open, &attr, pid, -1, -1, 0));
    }
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s COMMAND [ARGS...]\n", argv[0]);
        return 2;
    }

    // The child waits on the pipe until its counters are attached, so they see the exec.
    int gate[2];
    if (pipe(gate) != 0) {
        std::perror("pipe");
        return 2;
    }
    const pid_t child = fork();
    if (child < 0) {
        std::perror("fork");
        return 2;
    }
    if (child == 0) {
        close(gate[1]);
        char byte;
        if (read(gate[0], &byte, 1) != 1) {
            _exit(126);
        }
        close(gate[0]);
        execvp(argv[1], argv + 1);
        _exit(127);
    }
    close(gate[0]);

    TCounter counters[] = {
        {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
        {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
        {"branch-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
        {"L1d-misses", PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
        {"LLC-misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
    };
    for (TCounter& counter : counters) {
        counter.Fd = OpenCounter(counter, child);
    }
    const char go = 'x';
    if (write(gate[1], &go, 1) != 1) {
        std::perror("write");
    }
    close(gate[1]);

    int status = 0;
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
    for (const TCounter& counter : counters) {
        uint64_t value = 0;
        if (counter.Fd >= 0 && read(counter.Fd, &value, sizeof(value)) == sizeof(value)) {
            std::fprintf(stderr, "%s %llu\n", counter.Name, static_cast<unsigned long long>(value));
        }
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
