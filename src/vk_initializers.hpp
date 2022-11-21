// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.hpp>

namespace vkinit {

VkCommandPoolCreateInfo CommandPoolCreateInfo(
    uint32_t queue_family_index, VkCommandPoolCreateFlags flags = 0);

VkCommandBufferAllocateInfo CommandBufferAllocateInfo(
    const VkCommandPool& pool, uint32_t count = 1,
    VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

VkPipelineShaderStageCreateInfo PipelineShaderStageCreateInfo(
    VkShaderStageFlagBits stage, VkShaderModule module);

VkPipelineVertexInputStateCreateInfo VertexInputStateCreateInfo();

VkPipelineInputAssemblyStateCreateInfo InputAssemblyCreateInfo(
    VkPrimitiveTopology topology);

VkPipelineRasterizationStateCreateInfo RasterizationStateCreateInfo(
    VkPolygonMode mode);

VkPipelineMultisampleStateCreateInfo MultisamplingStateCreateInfo();

VkPipelineColorBlendAttachmentState ColorBlendAttachmentState();

VkPipelineLayoutCreateInfo PipelineLayoutCreateInfo();

VkImageCreateInfo ImageCreateInfo(VkFormat format,
                                  VkImageUsageFlags usage_flags,
                                  VkExtent3D extent);

VkImageViewCreateInfo ImageViewCreateInfo(VkFormat format, VkImage image,
                                          VkImageAspectFlags aspect_flags);

VkPipelineDepthStencilStateCreateInfo DepthStencilCreateInfo(
    bool depth_test, bool depth_write, VkCompareOp compare_op);

VkFenceCreateInfo FenceCreateInfo(VkFenceCreateFlags flags);

VkSemaphoreCreateInfo SemaphoreCreateInfo();

}  // namespace vkinit
