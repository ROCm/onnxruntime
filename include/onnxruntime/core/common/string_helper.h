// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>

// forward declaration
struct OrtAllocator;
namespace onnxruntime {
char* StrDup(const std::string& str, OrtAllocator* allocator);
wchar_t* StrDup(std::wstring_view str, OrtAllocator* allocator);
// Convert from UTF-8 string to wide string
wchar_t* StrDup(std::string_view str, OrtAllocator* allocator);
}  // namespace onnxruntime
