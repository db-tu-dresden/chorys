/**
 * \file granforge/dbops/limit_functor.hpp
 * \brief Lightweight offset/count tracker feeding the custom LIMIT ExecNode.
 *
 * The tracker retains just enough state to know which slice of each ExecBatch
 * should be forwarded downstream, allowing limits to be enforced without
 * materializing all rows in memory.
 */
#pragma once

#include <algorithm>
#include <cstdint>

namespace chorys::granforge::dbops {

/// \brief Tracks global LIMIT/OFFSET progress across batches.
class LimitTracker {
 public:
  /// \param offset Number of rows to skip prior to emitting results.
  /// \param count Maximum number of rows to produce.
  LimitTracker(int64_t offset, int64_t count)
      : offset_(std::max<int64_t>(0, offset)),
        count_(std::max<int64_t>(0, count)) {}

  /// \brief Decide which portion of a batch should be emitted.
  ///
  /// \param batch_length Length of the incoming batch.
  /// \param start Output start index relative to the batch.
  /// \param length Number of rows to emit from the batch.
  /// \return true when some portion should be forwarded downstream.
  bool SelectRange(int64_t batch_length, int64_t* start, int64_t* length) {
    if (count_ == 0 || produced_ >= count_) {
      consumed_ += batch_length;
      return false;
    }

    int64_t emit_start = 0;
    if (consumed_ < offset_) {
      int64_t remaining_offset = offset_ - consumed_;
      if (remaining_offset >= batch_length) {
        consumed_ += batch_length;
        return false;
      }
      emit_start = remaining_offset;
    }

    int64_t available = batch_length - emit_start;
    if (available <= 0) {
      consumed_ += batch_length;
      return false;
    }

    int64_t emit_len = std::min(count_ - produced_, available);
    consumed_ += batch_length;
    produced_ += emit_len;
    *start = emit_start;
    *length = emit_len;
    return emit_len > 0;
  }

  /// \brief Returns true once the requested number of rows has been produced.
  bool Done() const { return produced_ >= count_; }

 private:
  int64_t offset_;
  int64_t count_;
  int64_t consumed_ = 0;
  int64_t produced_ = 0;
};

}  // namespace chorys::granforge::dbops
