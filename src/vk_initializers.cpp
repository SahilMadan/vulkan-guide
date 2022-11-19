﻿#include <vk_initializers.hpp>

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

VkPipelineShaderStageCreateInfo PipelineShaderStageCreateInfo(
    VkShaderStageFlagBits stage, VkShaderModule module) {
  VkPipelineShaderStageCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  info.pNext = nullptr;

  info.stage = stage;
  info.module = module;
  info.pName = "main";

  return info;
}

VkPipelineVertexInputStateCreateInfo VertexInputStateCreateInfo() {
  VkPipelineVertexInputStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  info.pNext = nullptr;

  info.vertexBindingDescriptionCount = 0;
  info.vertexAttributeDescriptionCount = 0;

  return info;
}

VkPipelineInputAssemblyStateCreateInfo InputAssemblyCreateInfo(
    VkPrimitiveTopology topology) {
  VkPipelineInputAssemblyStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  info.pNext = nullptr;

  info.topology = topology;

  info.primitiveRestartEnable = VK_FALSE;

  return info;
}

VkPipelineRasterizationStateCreateInfo RasterizationStateCreateInfo(
    VkPolygonMode mode) {
  VkPipelineRasterizationStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  info.pNext = nullptr;

  info.depthClampEnable = VK_FALSE;
  // Discard all primitives before the rasterization stage (we don't want this).
  // Note: This would usually be used if we were reading from the buffer only.
  info.rasterizerDiscardEnable = VK_FALSE;

  info.polygonMode = mode;
  info.lineWidth = 1.f;

  // No backface culling.
  info.cullMode = VK_CULL_MODE_NONE;
  info.frontFace = VK_FRONT_FACE_CLOCKWISE;

  // No depth bias.
  info.depthBiasEnable = VK_FALSE;
  info.depthBiasConstantFactor = 0.f;
  info.depthBiasClamp = 0.f;
  info.depthBiasSlopeFactor = 0.f;

  return info;
}

VkPipelineMultisampleStateCreateInfo MultisamplingStateCreateInfo() {
  VkPipelineMultisampleStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  info.pNext = nullptr;

  info.sampleShadingEnable = VK_FALSE;
  // Multisampling defaulted to no multisampling (1 sample per pixel).
  info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  info.minSampleShading = 1.f;
  info.pSampleMask = nullptr;
  info.alphaToCoverageEnable = VK_FALSE;
  info.alphaToOneEnable = VK_FALSE;

  return info;
}

VkPipelineColorBlendAttachmentState ColorBlendAttachmentState() {
  VkPipelineColorBlendAttachmentState attachment = {};

  attachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  attachment.blendEnable = VK_FALSE;

  return attachment;
}

VkPipelineLayoutCreateInfo PipelineLayoutCreateInfo() {
  VkPipelineLayoutCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  info.pNext = nullptr;

  info.flags = 0;
  info.setLayoutCount = 0;
  info.pSetLayouts = nullptr;
  info.pushConstantRangeCount = 0;
  info.pPushConstantRanges = nullptr;

  return info;
}

VkImageCreateInfo ImageCreateInfo(VkFormat format,
                                  VkImageUsageFlags usage_flags,
                                  VkExtent3D extent) {
  VkImageCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.pNext = nullptr;

  info.imageType = VK_IMAGE_TYPE_2D;

  info.format = format;
  info.extent = extent;

  info.mipLevels = 1;
  // Used for layered textures. Not used here.
  info.arrayLayers = 1;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  // How the data for the texture is arranged in the GPU. Optimal allows for
  // effiency, using techniques like interleaving or swizzling, but won't allow
  // for the data to be read by the CPU.
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.usage = usage_flags;

  return info;
}

VkImageViewCreateInfo ImageViewCreateInfo(VkFormat format, VkImage image,
                                          VkImageAspectFlags aspect_flags) {
  VkImageViewCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  info.pNext = nullptr;

  info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  info.image = image;
  info.format = format;

  // SubresourceRange holds information about where the image points to, and is
  // used for layered images where you want to create an ImageView that points
  // to a specific layer. It's also possible to control mipmap levels.

  info.subresourceRange.baseMipLevel = 0;
  info.subresourceRange.levelCount = 1;
  info.subresourceRange.baseArrayLayer = 0;
  info.subresourceRange.layerCount = 1;
  info.subresourceRange.aspectMask = aspect_flags;

  return info;
}

VkPipelineDepthStencilStateCreateInfo DepthStencilCreateInfo(
    bool depth_test, bool depth_write, VkCompareOp compare_op) {
  VkPipelineDepthStencilStateCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  info.pNext = nullptr;

  info.depthTestEnable = depth_test ? VK_TRUE : VK_FALSE;
  info.depthWriteEnable = depth_write ? VK_TRUE : VK_FALSE;
  info.depthCompareOp = depth_test ? compare_op : VK_COMPARE_OP_ALWAYS;
  info.depthBoundsTestEnable = VK_FALSE;
  info.minDepthBounds = 1.f;
  info.maxDepthBounds = 1.f;
  info.stencilTestEnable = VK_FALSE;

  return info;
}

}  // namespace vkinit
