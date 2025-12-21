#pragma once

#include <concepts>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <vector>

namespace NProust {
    template <typename T> struct TTypeSerializer;

    template <typename T>
    concept Serializable = requires {
        sizeof(TTypeSerializer<T>);
    };

    template <typename T>
    requires std::is_trivially_copyable_v<T>
    struct TTypeSerializer<T> {
        static void Write(auto& writer, const T& value) {
            writer.WriteRaw(&value, sizeof(T));
        }

        static void Read(auto& reader, T& value) {
            reader.ReadRaw(&value, sizeof(T));
        }
    };

    template <typename T>
    requires Serializable<T>
    struct TTypeSerializer<std::vector<T>> {
        static void Write(auto& writer, const std::vector<T>& v) {
            size_t size = v.size();
            writer.WriteRaw(&size, sizeof(size));
            if constexpr (std::is_trivially_copyable_v<T>) {
                writer.WriteRaw(v.data(), sizeof(T) * size);
            } else {
                for (const auto& x : v) {
                    TTypeSerializer<T>::Write(writer, x);
                }
            }
        }

        static void Read(auto& reader, std::vector<T>& v) {
            size_t size;
            reader.ReadRaw(&size, sizeof(size));
            v.resize(size);
            if constexpr (std::is_trivially_copyable_v<T>) {
                reader.ReadRaw(v.data(), sizeof(T) * size);
            } else {
                for (auto& x : v) {
                    TTypeSerializer<T>::Read(reader, x);
                }
            }
        }
    };

    template <typename T>
    requires Serializable<T>
    struct TTypeSerializer<std::optional<T>> {
        static void Write(auto& writer, const std::optional<T>& v) {
            bool has = v.has_value();
            writer.WriteRaw(&has, sizeof(has));
            if (has) {
                TTypeSerializer<T>::Write(writer, *v);
            }
        }

        static void Read(auto& reader, std::optional<T>& v) {
            bool has;
            reader.ReadRaw(&has, sizeof(has));
            if (has) {
                T tmp;
                TTypeSerializer<T>::Read(reader, tmp);
                v = std::move(tmp);
            } else {
                v.reset();
            }
        }
    };
} // namespace NProust
