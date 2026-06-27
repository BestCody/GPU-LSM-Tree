#ifndef FLIX_NVTX3_STUB_HPP
#define FLIX_NVTX3_STUB_HPP

#include <string>

namespace nvtx3 {
template <typename Domain>
class scoped_range_in {
public:
  explicit scoped_range_in(const char*) {}
  explicit scoped_range_in(const std::string&) {}
};
}  // namespace nvtx3

#endif
