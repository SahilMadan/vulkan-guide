#include <vk_initializers.h>

namespace vkinit {

VkCommandPoolCreateInfo CommandPoolCreateInfo(uint32_t queue_family_index,
                                              VkCommandPoolCreateFlags flags) {
  // Create a command pool for submitting commands to the graphics queue. I.e.
  // object that command buffer memory is allocated from.
  VkCommandPoolCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  info.pNext = nullptr;

  info.queueFamilyIndex = queue_family_index;
  info.flags = flags;

  return info;
}

VkCommandBufferAllocateInfo CommandBufferAllocateInfo(
    const VkCommandPool& pool, uint32_t count, VkCommandBufferLevel level) {
  // Alloate the default command buffer that we will use for rendering.
  VkCommandBufferAllocateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  info.pNext = nullptr;

  info.commandPool = pool;
  info.commandBufferCount = count;
  // Primary buffers are sent to the queue, while secondary buffers act as
  // "subcommands" to a primary buffer.
  info.level = level;

  return info;
}

}  // namespace vkinit
