#pragma once

#include <deque>
#include <functional>

class DeletionQueue {
 public:
  void Push(std::function<void()>&& function);
  void Flush();

 private:
  std::deque<std::function<void()>> deletors_;
};
