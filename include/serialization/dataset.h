#pragma once

#include <fstream>

#include <spdlog/spdlog.h>

#include "../common/query.h"

// TODO: This code won't work correctly if the endianness of the platform that was 
// serializing the dataset and the one deserializing it are different. But I deliberately
// don't fix it: the serialize-deserialize are done only run the benchmarks, so
// it's evident that the dataset preparation and the test run itself will be done on the 
// same worker, so no need to overcomplicate. I check it though, to avoid surprises.
namespace NProust {
    inline constexpr uint32_t MAGIC_HEADER = 0x19101995;
    inline constexpr uint32_t CURRENT_VERSION = 1;
    inline constexpr size_t BUFFER_SIZE = 1 << 20; // 1 MB

    template <typename T>
    concept TriviallySerializable = std::is_trivially_copyable_v<T>;

    inline bool IsLittleEndianPlatform() {
        return std::endian::native == std::endian::little;
    }

    // Streamed dataset writer.
    //
    // The data types must be trivially copyable to be memcpy-safe and avoid UB. 
    // We use memcpy, and expect that the write will be vectorized by AVX/SSE.
    template<TriviallySerializable TValue, TriviallySerializable TOperationResult>
    class TStreamingDatasetWriter {
        using TItem = TTestPlanItem<TValue, TOperationResult>;

    public:
        // This is supposed to be the main use case
        TStreamingDatasetWriter(const size_t queryCount, const size_t length, const std::string& targetPath)
            : QueryCount(queryCount)
            , Length(length)
            , Buffer(BUFFER_SIZE)
            , OwnedOutput(std::make_unique<std::ofstream>(targetPath, std::ios::binary))
            , Output(*OwnedOutput)
        {
            if (OwnedOutput) {
                assert(OwnedOutput->is_open() && "Failed to create an output file");
            }
            WriteHeader();
        }

        // Just in case we want to stream it elsewhere over ostream interface
        TStreamingDatasetWriter(const size_t queryCount, const size_t length, std::ostream& externalWriter)
            : QueryCount(queryCount)
            , Length(length)
            , Buffer(BUFFER_SIZE)
            , Output(externalWriter)
            , OwnedOutput(nullptr)
        {
            WriteHeader();
        }

        void SaveQuery(const TItem& item) {
            assert(QueriesProcessed++ < QueryCount && "The number of queries exceeds the initially specified one");

            uint8_t operation = static_cast<uint8_t>(item.Operation);
            WriteRaw(&operation, sizeof(operation));
            WriteRaw(&item.Left, sizeof(item.Left));
            WriteRaw(&item.Right, sizeof(item.Right));

            const bool hasValue = item.Value.has_value();
            WriteRaw(&hasValue, sizeof(hasValue));
            if (hasValue) {
                WriteRaw(&item.Value.value(), sizeof(TValue));
            }

            const bool hasExpectedResult = item.ExpectedResult.has_value();
            WriteRaw(&hasExpectedResult, sizeof(hasExpectedResult));
            if (hasExpectedResult) {
                WriteRaw(&item.ExpectedResult.value(), sizeof(TOperationResult));
            }

            if (QueriesProcessed == QueryCount) {
                FlushBuffer(true);
            }
        }

        void Finalize() {
            if (QueriesProcessed != QueryCount) {
                spdlog::error("Query count mismatch: {} declared, {} factually serialized", QueriesProcessed, QueryCount);
            }
            if (BufferWritePosition > 0) {
                spdlog::warn("Flushing the buffer manually, which sholdn't normally happen in Finalize.");
                FlushBuffer(true);
            }
        }

    private:
        const size_t QueryCount;
        const size_t Length;
        std::ostream& Output;
        std::unique_ptr<std::ofstream> OwnedOutput;

        size_t QueriesProcessed = 0;

        // Manual buffer maintenance is better, although isn't a must here.
        // We control the chunk size (1 MB), so less write(2) syscalls,
        // and have much less virtual method calls for underlying ostream&
        // interface.
        std::vector<char> Buffer;
        size_t BufferWritePosition = 0;

        void WriteHeader() {
            WriteRaw(&MAGIC_HEADER, sizeof(MAGIC_HEADER));
            WriteRaw(&CURRENT_VERSION, sizeof(CURRENT_VERSION));
            const bool isLittleEndian = IsLittleEndianPlatform();
            WriteRaw(&isLittleEndian, sizeof(isLittleEndian));

            WriteRaw(&Length, sizeof(Length));
            WriteRaw(&QueryCount, sizeof(QueryCount));
        }

        void WriteRaw(const void* data, size_t sliceLength) {
            const char* dataPtr = (const char*)data;
            while (sliceLength > 0) {
                size_t remaining = Buffer.size() - BufferWritePosition;
                size_t currentChunkSize = std::min(remaining, sliceLength);
                std::memcpy(Buffer.data() + BufferWritePosition, dataPtr, currentChunkSize);
                BufferWritePosition += currentChunkSize;
                dataPtr += currentChunkSize;
                sliceLength -= currentChunkSize;

                if (BufferWritePosition == Buffer.size()) {
                    FlushBuffer(false);
                }
            }
        }

        void FlushBuffer(const bool isForced) {
            Output.write(Buffer.data(), BufferWritePosition);
            BufferWritePosition = 0;
            if (isForced) {
                Output.flush();
            }
            assert(Output.good() && "Stream record has failed");
        }
    };

    // Reader, mostly mirroring the implementation of TStreamingDatasetWriter
    template<TriviallySerializable TValue, TriviallySerializable TOperationResult>
    class TStreamingDatasetReader {
        using TItem = TTestPlanItem<TValue, TOperationResult>;

    public:
        TStreamingDatasetReader(const std::string& path)
            : Buffer(BUFFER_SIZE)
        {
            OwnedInput = std::make_unique<std::ifstream>(path, std::ios::binary);
            Input = OwnedInput.get();
            assert(static_cast<std::ifstream*>(Input)->is_open());

            ReadHeader();
        }

        TStreamingDatasetReader(std::istream& externalReader)
            : Input(externalReader)
            , OwnedInput(nullptr)
            , Buffer(BUFFER_SIZE)
        {
            ReadHeader();
        }

        size_t GetLength() const {
            return Length;
        }

        std::optional<TItem> NextQuery() {
            if (QueriesRead++ >= QueryCount) {
                return std::nullopt;
            }

            TItem item;

            uint8_t operation;
            ReadRaw(&operation, sizeof(operation));
            item.Operation = static_cast<EEngineOperation>(operation);

            ReadRaw(&item.Left, sizeof(item.Left));
            ReadRaw(&item.Right, sizeof(item.Right));

            bool hasValue;
            ReadRaw(&hasValue, sizeof(hasValue));
            if (hasValue) {
                TValue value;
                ReadRaw(&value, sizeof(TValue));
                item.Value = value;
            } else {
                item.Value.reset();
            }

            bool hasExpectedResult;
            ReadRaw(&hasExpectedResult, sizeof(hasExpectedResult));
            if (hasExpectedResult) {
                TOperationResult expectedResult;
                ReadRaw(&expectedResult, sizeof(TOperationResult));
                item.ExpectedResult = expectedResult;
            } else {
                item.ExpectedResult.reset();
            }

            return item;
        }

    private:
        size_t QueryCount;
        size_t Length;
        size_t QueriesRead = 0;

        std::istream& Input;
        std::unique_ptr<std::istream> OwnedInput;

        std::vector<char> Buffer;
        size_t BufferSize = 0;
        size_t BufferReadPosition = 0;

        void ReadHeader() {
            uint32_t magic;
            uint32_t version;
            bool isLittleEndianFormat;

            ReadRaw(&magic, sizeof(magic));
            ReadRaw(&version, sizeof(version));
            ReadRaw(&isLittleEndianFormat, sizeof(isLittleEndianFormat));
            ReadRaw(&Length, sizeof(Length));
            ReadRaw(&QueryCount, sizeof(QueryCount));

            if (magic != MAGIC_HEADER || version != CURRENT_VERSION || isLittleEndianFormat != IsLittleEndianPlatform()) {
                // We don't trust external input, so we're stricter about the check and do the std::terminate.

                spdlog::error("Invalid dataset header");
                std::terminate();
            }
        }

        void ReadRaw(void* dst, size_t sliceLength) {
            char* target = reinterpret_cast<char*>(dst);
            while (sliceLength > 0) {
                if (BufferReadPosition == BufferSize) {
                    FillBuffer();
                }

                size_t available = BufferSize - BufferReadPosition;
                size_t toCopy = std::min(sliceLength, available);
                std::memcpy(target, Buffer.data() + BufferReadPosition, toCopy);

                BufferReadPosition += toCopy;
                target += toCopy;
                sliceLength -= toCopy;
            }
        }

        void FillBuffer() {
            BufferReadPosition = 0;
            Input.read(reinterpret_cast<char*>(Buffer.data()), Buffer.size());

            BufferSize = static_cast<size_t>(Input.gcount());
            if (BufferSize == 0 && QueriesRead < QueryCount) {
                std::cerr << "Unexpected end of file" << std::endl;
                std::terminate();
            }
        }
    };
}
