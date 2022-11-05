#include "deletion_queue.hpp"

void DeletionQueue::Push(std::function<void()>&& function) {
  deletors_.push_back(function);
}

void DeletionQueue::Flush() {
  for (auto it = deletors_.rbegin(); it != deletors_.rend(); it++) {
    (*it)();
  }

  deletors_.clear();
}