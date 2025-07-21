// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>

// forward declaration
struct OrtAllocator;
namespace onnxruntime {
char* StrDup(std::string_view str, OrtAllocator* allocator);
inline char* StrDup(const std::string& str, OrtAllocator* allocator) {
  return StrDup(std::string_view{str}, allocator);
}
wchar_t* StrDup(std::wstring_view str, OrtAllocator* allocator);
// Convert from UTF-8 string to wide string
void StrConvert(std::string_view str, wchar_t* &dst, OrtAllocator* allocator);
inline void StrConvert(std::string_view str, char* &dst, OrtAllocator* allocator) {
  dst = StrDup(str, allocator);
}

}  // namespace onnxruntime
