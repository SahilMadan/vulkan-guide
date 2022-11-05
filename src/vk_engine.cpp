﻿
#include "vk_engine.hpp"

#include <SDL.h>
#include <SDL_vulkan.h>
#include <VkBootstrap.h>
#include <vk_initializers.hpp>
#include <vk_types.hpp>

#include <fstream>
#include <iostream>

namespace {

constexpr uint32_t kSyncWaitTimeoutNs = 1000000000;

// Immediately abort on error.
#define VK_CHECK(x)                                               \
  do {                                                            \
    VkResult err = x;                                             \
    if (err) {                                                    \
      std::cout << "Detected Vulkan error: " << err << std::endl; \
      abort();                                                    \
    }                                                             \
  } while (0)

}  // namespace

void VulkanEngine::Init() {
  // We initialize SDL and create a window with it.
  SDL_Init(SDL_INIT_VIDEO);

  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

  window_ = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED,
                             SDL_WINDOWPOS_UNDEFINED, window_extent_.width,
                             window_extent_.height, window_flags);

  // everything went fine
  is_initialized_ = true;

  InitVulkan();
  InitSwapchain();
  InitCommands();
  InitDefaultRenderpass();
  InitFramebuffers();
  InitSyncStructs();
  InitPipelines();
}

void VulkanEngine::InitVulkan() {
  // Make Vulkan instance with basic debug features.
  vkb::InstanceBuilder builder;
  auto instance_result = builder.set_app_name("VkRenderer")
                             .request_validation_layers(true)
                             .require_api_version(1, 1, 0)
                             .use_default_debug_messenger()
                             .build();

  vkb::Instance vkb_instance = instance_result.value();
  instance_ = vkb_instance.instance;
  debug_messenger_ = vkb_instance.debug_messenger;

  // Get the surface of the window we opened with SDL.
  SDL_Vulkan_CreateSurface(window_, instance_, &surface_);

  // Select a GPU (one which we can write to our surface and supports
  // Vulkan 1.1).
  vkb::PhysicalDeviceSelector selector{vkb_instance};
  vkb::PhysicalDevice vkb_physical_device =
      selector.set_minimum_version(1, 1).set_surface(surface_).select().value();

  // Create the Vulkan device.
  vkb::DeviceBuilder device_builder{vkb_physical_device};
  vkb::Device vkb_device = device_builder.build().value();

  device_ = vkb_device.device;
  gpu_ = vkb_physical_device.physical_device;

  // Also grab the graphics queue (i.e. the execution port on the GPU).
  graphics_queue_ = vkb_device.get_queue(vkb::QueueType::graphics).value();
  graphics_queue_family_ =
      vkb_device.get_queue_index(vkb::QueueType::graphics).value();
}

void VulkanEngine::InitSwapchain() {
  vkb::SwapchainBuilder swapchain_builder(gpu_, device_, surface_);

  vkb::Swapchain vkb_swapchain =
      swapchain_builder.use_default_format_selection()
          .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
          .set_desired_extent(window_extent_.width, window_extent_.height)
          .build()
          .value();

  swapchain_ = vkb_swapchain.swapchain;
  swapchain_images_ = vkb_swapchain.get_images().value();
  swapchain_image_views_ = vkb_swapchain.get_image_views().value();
  swapchain_image_format_ = vkb_swapchain.image_format;

  deletion_queue_.Push([=]() {
    vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    for (int i = 0; i < swapchain_image_views_.size(); i++) {
      vkDestroyFramebuffer(device_, framebuffers_[i], nullptr);

      vkDestroyImageView(device_, swapchain_image_views_[i], nullptr);
    }
  });
}

void VulkanEngine::InitCommands() {
  // Create a command pool for submitting commands to the graphics queue. I.e.
  // object that command buffer memory is allocated from.
  VkCommandPoolCreateInfo command_pool_info = vkinit::CommandPoolCreateInfo(
      graphics_queue_family_, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  VK_CHECK(vkCreateCommandPool(device_, &command_pool_info, nullptr,
                               &command_pool_));

  deletion_queue_.Push([=]() {
    // This will also destroy all allocated command buffers.
    vkDestroyCommandPool(device_, command_pool_, nullptr);
  });

  // Alloate the default command buffer that we will use for rendering.
  VkCommandBufferAllocateInfo command_buffer_allocate_info =
      vkinit::CommandBufferAllocateInfo(command_pool_);

  VK_CHECK(vkAllocateCommandBuffers(device_, &command_buffer_allocate_info,
                                    &command_buffer_));
}

void VulkanEngine::InitDefaultRenderpass() {
  // The renderpass will use this color attachment.
  VkAttachmentDescription color_attachment = {};
  color_attachment.format = swapchain_image_format_;
  // 1 sample -- we won't be doing MSAA.
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  // Clear when the attachment is loaded.
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  // Keep stored when the renderpass ends.
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  // We don't care about stencil.
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  // We don't know/care about the starting layout.
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  // After renderpass ends, layout should be ready for display.
  color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  // Add a subpass that will render into it.

  VkAttachmentReference color_attachment_ref = {};
  // Attachment number will indedx into the pAttachments array in the parent
  // renderpass.
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  // We are going to create 1 subpass (minimum) for our renderpass.
  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;

  // Create the renderpass (and attach the subpass).

  VkRenderPassCreateInfo renderpass_info = {};
  renderpass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderpass_info.pNext = nullptr;

  renderpass_info.attachmentCount = 1;
  renderpass_info.pAttachments = &color_attachment;
  renderpass_info.subpassCount = 1;
  renderpass_info.pSubpasses = &subpass;

  VK_CHECK(
      vkCreateRenderPass(device_, &renderpass_info, nullptr, &renderpass_));

  deletion_queue_.Push(
      [=]() { vkDestroyRenderPass(device_, renderpass_, nullptr); });
}

void VulkanEngine::InitFramebuffers() {
  // Create the framebuffers for the swapchain images. This will connect the
  // renderpass to the images for rendering.
  VkFramebufferCreateInfo framebuffer_info = {};
  framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_info.pNext = nullptr;

  framebuffer_info.renderPass = renderpass_;
  framebuffer_info.attachmentCount = 1;
  framebuffer_info.width = window_extent_.width;
  framebuffer_info.height = window_extent_.height;
  framebuffer_info.layers = 1;

  uint32_t swapchain_image_count = swapchain_images_.size();
  framebuffers_ = std::vector<VkFramebuffer>(swapchain_image_count);

  // Create a framebuffer for each of the swapchain image views.
  for (int i = 0; i < swapchain_image_count; i++) {
    framebuffer_info.pAttachments = &swapchain_image_views_[i];
    VK_CHECK(vkCreateFramebuffer(device_, &framebuffer_info, nullptr,
                                 &framebuffers_[i]));
  }
}

void VulkanEngine::InitSyncStructs() {
  VkFenceCreateInfo fence_info = {};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.pNext = nullptr;

  // Signal when created so we can wait on it before using it on the GPU.
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  VK_CHECK(vkCreateFence(device_, &fence_info, nullptr, &render_fence_));

  deletion_queue_.Push(
      [=]() { vkDestroyFence(device_, render_fence_, nullptr); });

  VkSemaphoreCreateInfo semaphore_info = {};
  semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_info.pNext = nullptr;

  semaphore_info.flags = 0;

  VK_CHECK(vkCreateSemaphore(device_, &semaphore_info, nullptr,
                             &present_semaphore_));
  deletion_queue_.Push(
      [=]() { vkDestroySemaphore(device_, present_semaphore_, nullptr); });

  VK_CHECK(
      vkCreateSemaphore(device_, &semaphore_info, nullptr, &render_semaphore_));
  deletion_queue_.Push(
      [=]() { vkDestroySemaphore(device_, render_semaphore_, nullptr); });
}

void VulkanEngine::InitPipelines() {
  auto triangle_frag_shader =
      LoadShaderModule("../../shaders/triangle.frag.spv");
  if (triangle_frag_shader.has_value()) {
    std::cout << "Triangle fragment shader successfully loaded.\n";
  } else {
    std::cerr << "Error when building triangle fragment shader.\n";
  }

  auto triangle_vert_shader =
      LoadShaderModule("../../shaders/triangle.vert.spv");
  if (triangle_vert_shader.has_value()) {
    std::cout << "Triangle vertex shader successfully loaded.\n";
  } else {
    std::cerr << "Error when building triangle vertex shader.\n";
  }

  VkPipelineLayoutCreateInfo pipeline_layout_info =
      vkinit::PipelineLayoutCreateInfo();
  VK_CHECK(vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr,
                                  &triangle_pipeline_layout_));

  deletion_queue_.Push([=]() {
    vkDestroyPipelineLayout(device_, triangle_pipeline_layout_, nullptr);
  });

  PipelineBuilder builder;

  // Shader stages.
  builder.shader_stages.push_back(vkinit::PipelineShaderStageCreateInfo(
      VK_SHADER_STAGE_VERTEX_BIT, triangle_vert_shader.value()));
  builder.shader_stages.push_back(vkinit::PipelineShaderStageCreateInfo(
      VK_SHADER_STAGE_FRAGMENT_BIT, triangle_frag_shader.value()));

  // How to read vertices from vertex buffers.
  builder.vertex_input_info = vkinit::VertexInputStateCreateInfo();

  builder.input_assembly =
      vkinit::InputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

  builder.viewport.x = 0.f;
  builder.viewport.y = 0.f;
  builder.viewport.width = static_cast<float>(window_extent_.width);
  builder.viewport.height = static_cast<float>(window_extent_.height);
  builder.viewport.minDepth = 0.f;
  builder.viewport.maxDepth = 1.f;

  builder.scissor.offset = {0, 0};
  builder.scissor.extent = window_extent_;

  builder.rasterizer =
      vkinit::RasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);

  builder.multisampling = vkinit::MultisamplingStateCreateInfo();

  builder.color_blend_attachment = vkinit::ColorBlendAttachmentState();

  builder.pipeline_layout = triangle_pipeline_layout_;

  triangle_pipeline_ = builder.BuildPipeline(device_, renderpass_);

  deletion_queue_.Push(
      [=]() { vkDestroyPipeline(device_, triangle_pipeline_, nullptr); });
  vkDestroyShaderModule(device_, triangle_vert_shader.value(), nullptr);
  vkDestroyShaderModule(device_, triangle_frag_shader.value(), nullptr);

  auto colored_triangle_frag_shader =
      LoadShaderModule("../../shaders/colored_triangle.frag.spv");
  if (colored_triangle_frag_shader.has_value()) {
    std::cout << "Colored triangle fragment shader successfully loaded.\n";
  } else {
    std::cerr << "Error when building colored triangle fragment shader.\n";
  }

  auto colored_triangle_vert_shader =
      LoadShaderModule("../../shaders/colored_triangle.vert.spv");
  if (triangle_vert_shader.has_value()) {
    std::cout << "Colored triangle vertex shader successfully loaded.\n";
  } else {
    std::cerr << "Error when building colored triangle vertex shader.\n";
  }

  builder.shader_stages.clear();
  builder.shader_stages.push_back(vkinit::PipelineShaderStageCreateInfo(
      VK_SHADER_STAGE_VERTEX_BIT, colored_triangle_vert_shader.value()));
  builder.shader_stages.push_back(vkinit::PipelineShaderStageCreateInfo(
      VK_SHADER_STAGE_FRAGMENT_BIT, colored_triangle_frag_shader.value()));

  colored_triangle_pipeline_ = builder.BuildPipeline(device_, renderpass_);

  deletion_queue_.Push([=]() {
    vkDestroyPipeline(device_, colored_triangle_pipeline_, nullptr);
  });
  vkDestroyShaderModule(device_, colored_triangle_vert_shader.value(), nullptr);
  vkDestroyShaderModule(device_, colored_triangle_frag_shader.value(), nullptr);
}

std::optional<VkShaderModule> VulkanEngine::LoadShaderModule(
    const std::string& path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    return std::nullopt;
  }

  // Already at end of file (std::ios::ate).
  size_t file_size_bytes = static_cast<size_t>(file.tellg());

  // Spirv expects the buffer to be uint32.
  std::vector<uint32_t> buffer(file_size_bytes / sizeof(uint32_t));

  file.seekg(0);

  file.read(reinterpret_cast<char*>(buffer.data()), file_size_bytes);

  file.close();

  VkShaderModuleCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.pNext = nullptr;

  create_info.codeSize = file_size_bytes;
  create_info.pCode = buffer.data();

  VkShaderModule shader_module;
  if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) !=
      VK_SUCCESS) {
    return std::nullopt;
  }

  return shader_module;
}

void VulkanEngine::Cleanup() {
  if (is_initialized_) {
    // Wait for VkCommandBuffer to finish before destroying render pass.
    VK_CHECK(
        vkWaitForFences(device_, 1, &render_fence_, true, kSyncWaitTimeoutNs));

    deletion_queue_.Flush();

    vkDestroyDevice(device_, nullptr);
    vkDestroySurfaceKHR(instance_, surface_, nullptr);
    vkb::destroy_debug_utils_messenger(instance_, debug_messenger_);
    vkDestroyInstance(instance_, nullptr);

    SDL_DestroyWindow(window_);
  }
}

void VulkanEngine::Draw() {
  // Wait until the GPU has finished rendering the last frame. Timeout of 1
  // second.
  VK_CHECK(
      vkWaitForFences(device_, 1, &render_fence_, true, kSyncWaitTimeoutNs));
  VK_CHECK(vkResetFences(device_, 1, &render_fence_));

  // Request image from the swapchain, one second timeout.
  uint32_t swapchain_image_index;
  VK_CHECK(vkAcquireNextImageKHR(device_, swapchain_, kSyncWaitTimeoutNs,
                                 present_semaphore_, nullptr,
                                 &swapchain_image_index));

  // Now that we're sure that the commands have finished executing, we can
  // safely reset the command buffer to begin recording again.
  VK_CHECK(vkResetCommandBuffer(command_buffer_, 0));

  // Begin command buffer recording. We will use this command buffer exactly
  // once, so we want to let Vulkan know that.
  VkCommandBufferBeginInfo command_begin_info = {};
  command_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  command_begin_info.pNext = nullptr;

  command_begin_info.pInheritanceInfo = nullptr;
  command_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  VK_CHECK(vkBeginCommandBuffer(command_buffer_, &command_begin_info));

  // DRAW

  VkClearValue clear_value;
  float flash = abs(sin(framenumber_ / 120.f));
  clear_value.color = {{0.0f, 0.0f, flash, 1.0f}};

  // Main renderpass.
  VkRenderPassBeginInfo renderpass_begin_info = {};
  renderpass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderpass_begin_info.pNext = nullptr;

  renderpass_begin_info.renderPass = renderpass_;
  renderpass_begin_info.renderArea.offset.x = 0;
  renderpass_begin_info.renderArea.offset.y = 0;
  renderpass_begin_info.renderArea.extent = window_extent_;
  renderpass_begin_info.framebuffer = framebuffers_[swapchain_image_index];

  renderpass_begin_info.clearValueCount = 1;
  renderpass_begin_info.pClearValues = &clear_value;

  vkCmdBeginRenderPass(command_buffer_, &renderpass_begin_info,
                       VK_SUBPASS_CONTENTS_INLINE);

  // Mesh draws would go here...
  switch (selected_shader_) {
    case 0:
      vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        colored_triangle_pipeline_);
      break;
    case 1:
      vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        triangle_pipeline_);
      break;
    default:
      std::cerr << "Unexpected selected shader.\n";
  }

  vkCmdDraw(command_buffer_, 3, 1, 0, 0);

  // Finalize the render pass and command buffer.
  vkCmdEndRenderPass(command_buffer_);
  VK_CHECK(vkEndCommandBuffer(command_buffer_));

  // Prepare the submission to the queue.
  // We want to wait on the |present_semaphore_|, as that semaphore is signaled
  // when the swapchain is ready. We will use the |render_semaphore_| when
  // rendering has finish.
  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.pNext = nullptr;

  VkPipelineStageFlags wait_stage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

  submit_info.pWaitDstStageMask = &wait_stage;

  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = &present_semaphore_;

  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = &render_semaphore_;

  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer_;

  // Submit command buffer to the queue and execute it.
  // |render_fence_| will now block until the graphic command finishes.
  VK_CHECK(vkQueueSubmit(graphics_queue_, 1, &submit_info, render_fence_));

  // DISPLAY

  // THis will put the image we just rendered into the visible window.
  // We want to wait on the |render_semaphore_| for that, as it's necessary that
  // drawing commands have finished before the image is displayed to the user.
  VkPresentInfoKHR present_info = {};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.pNext = nullptr;

  present_info.swapchainCount = 1;
  present_info.pSwapchains = &swapchain_;

  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &render_semaphore_;

  present_info.pImageIndices = &swapchain_image_index;

  VK_CHECK(vkQueuePresentKHR(graphics_queue_, &present_info));

  framenumber_++;
}

void VulkanEngine::Run() {
  SDL_Event e;
  bool bQuit = false;

  // main loop
  while (!bQuit) {
    // Handle events on queue
    while (SDL_PollEvent(&e) != 0) {
      // close the window when user alt-f4s or clicks the X button
      if (e.type == SDL_QUIT) {
        bQuit = true;
      } else if (e.type == SDL_KEYDOWN) {
        if (e.key.keysym.sym == SDLK_SPACE) {
          selected_shader_ = (selected_shader_ + 1) % 2;
        }
      }
    }

    Draw();
  }
}

VkPipeline VulkanEngine::PipelineBuilder::BuildPipeline(
    VkDevice device, VkRenderPass renderpass) {
  VkPipelineViewportStateCreateInfo viewport_state = {};
  viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewport_state.pNext = nullptr;

  viewport_state.viewportCount = 1;
  viewport_state.pViewports = &viewport;
  viewport_state.scissorCount = 1;
  viewport_state.pScissors = &scissor;

  VkPipelineColorBlendStateCreateInfo color_blending = {};
  color_blending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  color_blending.pNext = nullptr;

  color_blending.logicOpEnable = VK_FALSE;
  color_blending.logicOp = VK_LOGIC_OP_COPY;
  color_blending.attachmentCount = 1;
  color_blending.pAttachments = &color_blend_attachment;

  VkGraphicsPipelineCreateInfo pipeline_info = {};
  pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipeline_info.pNext = nullptr;

  pipeline_info.stageCount = shader_stages.size();
  pipeline_info.pStages = shader_stages.data();
  pipeline_info.pVertexInputState = &vertex_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state;
  pipeline_info.pRasterizationState = &rasterizer;
  pipeline_info.pMultisampleState = &multisampling;
  pipeline_info.pColorBlendState = &color_blending;
  pipeline_info.layout = pipeline_layout;
  pipeline_info.renderPass = renderpass;
  pipeline_info.subpass = 0;
  pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

  VkPipeline pipeline;
  if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                nullptr, &pipeline) != VK_SUCCESS) {
    std::cerr << "Failed to create pipeline\n";
    return VK_NULL_HANDLE;
  }

  return pipeline;
}
