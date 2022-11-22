
#include "vk_engine.hpp"

#include <SDL.h>
#include <SDL_vulkan.h>
#include <VkBootstrap.h>

#include <fstream>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <vk_initializers.hpp>
#include <vk_types.hpp>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace {

constexpr uint32_t kSyncWaitTimeoutNs = 1'000'000'000;
constexpr uint32_t kUploadSyncWaitTimeoutNs = 9'999'999'999;

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

  InitVulkan();
  InitSwapchain();
  InitCommands();
  InitDefaultRenderpass();
  InitFramebuffers();
  InitSyncStructs();
  InitDescriptors();
  InitPipelines();

  LoadTextures();
  LoadMeshes();

  InitScene();

  // everything went fine
  is_initialized_ = true;
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

  VkPhysicalDeviceShaderDrawParametersFeatures draw_parameters_features = {};
  draw_parameters_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES;
  draw_parameters_features.pNext = nullptr;

  draw_parameters_features.shaderDrawParameters = VK_TRUE;

  vkb::Device vkb_device =
      device_builder.add_pNext(&draw_parameters_features).build().value();

  device_ = vkb_device.device;
  gpu_ = vkb_physical_device.physical_device;

  // Also grab the graphics queue (i.e. the execution port on the GPU).
  graphics_queue_ = vkb_device.get_queue(vkb::QueueType::graphics).value();
  graphics_queue_family_ =
      vkb_device.get_queue_index(vkb::QueueType::graphics).value();

  // Initialize the memory allocator.
  VmaAllocatorCreateInfo allocator_info = {};
  allocator_info.physicalDevice = gpu_;
  allocator_info.device = device_;
  allocator_info.instance = instance_;
  vmaCreateAllocator(&allocator_info, &allocator_);

  deletion_queue_.Push([=]() { vmaDestroyAllocator(allocator_); });

  gpu_properties_ = vkb_device.physical_device.properties;
  std::cout << "The GPU has a minimum buffer alignment of: "
            << gpu_properties_.limits.minUniformBufferOffsetAlignment
            << std::endl;
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

  VkExtent3D depth_image_extent = {window_extent_.width, window_extent_.height,
                                   1};

  // Hardcode the depth format to 32 bit float.
  depth_format_ = VK_FORMAT_D32_SFLOAT;

  // Create a depth image with the depth attachment usage flag.
  VkImageCreateInfo depth_image_info = vkinit::ImageCreateInfo(
      depth_format_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      depth_image_extent);

  VmaAllocationCreateInfo depth_image_allocation_info = {};
  depth_image_allocation_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  depth_image_allocation_info.requiredFlags =
      VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  vmaCreateImage(allocator_, &depth_image_info, &depth_image_allocation_info,
                 &depth_image_.image, &depth_image_.allocation, nullptr);

  VkImageViewCreateInfo depth_image_view_info = vkinit::ImageViewCreateInfo(
      depth_format_, depth_image_.image, VK_IMAGE_ASPECT_DEPTH_BIT);

  VK_CHECK(vkCreateImageView(device_, &depth_image_view_info, nullptr,
                             &depth_image_view_));

  deletion_queue_.Push([=]() {
    vkDestroyImageView(device_, depth_image_view_, nullptr);
    vmaDestroyImage(allocator_, depth_image_.image, depth_image_.allocation);
  });
}

void VulkanEngine::InitCommands() {
  // Create a command pool for submitting commands to the graphics queue. I.e.
  // object that command buffer memory is allocated from.
  VkCommandPoolCreateInfo command_pool_info = vkinit::CommandPoolCreateInfo(
      graphics_queue_family_, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  for (int i = 0; i < kVkFrameOverlap; i++) {
    VK_CHECK(vkCreateCommandPool(device_, &command_pool_info, nullptr,
                                 &frames_[i].command_pool));

    deletion_queue_.Push([=]() {
      // This will also destroy all allocated command buffers.
      vkDestroyCommandPool(device_, frames_[i].command_pool, nullptr);
    });

    // Alloate the default command buffer that we will use for rendering.
    VkCommandBufferAllocateInfo command_buffer_allocate_info =
        vkinit::CommandBufferAllocateInfo(frames_[i].command_pool);

    VK_CHECK(vkAllocateCommandBuffers(device_, &command_buffer_allocate_info,
                                      &frames_[i].command_buffer));
  }

  VkCommandPoolCreateInfo upload_command_pool =
      vkinit::CommandPoolCreateInfo(graphics_queue_family_);

  VK_CHECK(vkCreateCommandPool(device_, &upload_command_pool, nullptr,
                               &upload_context_.command_pool));
  deletion_queue_.Push([=]() {
    vkDestroyCommandPool(device_, upload_context_.command_pool, nullptr);
  });

  // Alloate the default command buffer that we will use for instant commands.
  VkCommandBufferAllocateInfo upload_buffer_allocate_info =
      vkinit::CommandBufferAllocateInfo(upload_context_.command_pool, 1);

  VK_CHECK(vkAllocateCommandBuffers(device_, &upload_buffer_allocate_info,
                                    &upload_context_.command_buffer));
}

void VulkanEngine::InitDefaultRenderpass() {
  // The renderpass will use this color attachment.
  VkAttachmentDescription color_attachment = {};
  color_attachment.flags = 0;
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

  VkAttachmentDescription depth_attachment = {};
  depth_attachment.flags = 0;
  depth_attachment.format = depth_format_;
  depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depth_attachment.finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depth_attachment_ref = {};
  depth_attachment_ref.attachment = 1;
  depth_attachment_ref.layout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  // We are going to create 1 subpass (minimum) for our renderpass.
  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;
  subpass.pDepthStencilAttachment = &depth_attachment_ref;

  VkSubpassDependency dependency = {};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  // Tell Vulkan that the depth attachment in a renderpass cannot be used before
  // previous renderpasses have finished.
  VkSubpassDependency depth_dependency = {};
  depth_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  depth_dependency.dstSubpass = 0;
  depth_dependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  depth_dependency.srcAccessMask = 0;
  depth_dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  depth_dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  // Create the renderpass (and attach the subpass).

  VkRenderPassCreateInfo renderpass_info = {};
  renderpass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderpass_info.pNext = nullptr;

  VkAttachmentDescription attachments[2] = {color_attachment, depth_attachment};
  renderpass_info.attachmentCount = 2;
  renderpass_info.pAttachments = &attachments[0];
  renderpass_info.subpassCount = 1;
  renderpass_info.pSubpasses = &subpass;

  VkSubpassDependency dependencies[2] = {dependency, depth_dependency};
  renderpass_info.dependencyCount = 2;
  renderpass_info.pDependencies = &dependencies[0];

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
    VkImageView attachments[2];
    attachments[0] = swapchain_image_views_[i];
    attachments[1] = depth_image_view_;

    framebuffer_info.pAttachments = attachments;
    framebuffer_info.attachmentCount = 2;

    VK_CHECK(vkCreateFramebuffer(device_, &framebuffer_info, nullptr,
                                 &framebuffers_[i]));
  }
}

void VulkanEngine::InitSyncStructs() {
  // Signal when created so we can wait on it before using it on the GPU.
  VkFenceCreateInfo fence_info =
      vkinit::FenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);

  VkSemaphoreCreateInfo semaphore_info = vkinit::SemaphoreCreateInfo();

  for (int i = 0; i < kVkFrameOverlap; i++) {
    VK_CHECK(
        vkCreateFence(device_, &fence_info, nullptr, &frames_[i].render_fence));

    deletion_queue_.Push(
        [=]() { vkDestroyFence(device_, frames_[i].render_fence, nullptr); });

    VK_CHECK(vkCreateSemaphore(device_, &semaphore_info, nullptr,
                               &frames_[i].present_semaphore));

    deletion_queue_.Push([=]() {
      vkDestroySemaphore(device_, frames_[i].present_semaphore, nullptr);
    });

    VK_CHECK(vkCreateSemaphore(device_, &semaphore_info, nullptr,
                               &frames_[i].render_semaphore));

    deletion_queue_.Push([=]() {
      vkDestroySemaphore(device_, frames_[i].render_semaphore, nullptr);
    });
  }

  VkFenceCreateInfo upload_fence_info = vkinit::FenceCreateInfo();
  VK_CHECK(vkCreateFence(device_, &upload_fence_info, nullptr,
                         &upload_context_.upload_fence));

  deletion_queue_.Push([=]() {
    vkDestroyFence(device_, upload_context_.upload_fence, nullptr);
  });
}

void VulkanEngine::InitDescriptors() {
  std::vector<VkDescriptorPoolSize> sizes = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10},
  };

  VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
  descriptor_pool_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptor_pool_create_info.pNext = nullptr;

  descriptor_pool_create_info.flags = 0;
  descriptor_pool_create_info.maxSets = 10;
  descriptor_pool_create_info.poolSizeCount =
      static_cast<uint32_t>(sizes.size());
  descriptor_pool_create_info.pPoolSizes = sizes.data();

  vkCreateDescriptorPool(device_, &descriptor_pool_create_info, nullptr,
                         &descriptor_pool_);

  // Binding for camera data at 0.
  VkDescriptorSetLayoutBinding camera_buffer_binding =
      vkinit::DescriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                         VK_SHADER_STAGE_VERTEX_BIT, 0);

  // Binding for scene data at 1.
  VkDescriptorSetLayoutBinding scene_buffer_binding =
      vkinit::DescriptorSetLayoutBinding(
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 1);

  VkDescriptorSetLayoutBinding bindings[] = {camera_buffer_binding,
                                             scene_buffer_binding};

  VkDescriptorSetLayoutCreateInfo set_info = {};
  set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  set_info.pNext = nullptr;

  set_info.bindingCount = 2;
  set_info.flags = 0;
  set_info.pBindings = bindings;

  vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &global_set_layout_);

  VkDescriptorSetLayoutBinding object_buffer_binding =
      vkinit::DescriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         VK_SHADER_STAGE_VERTEX_BIT, 0);

  VkDescriptorSetLayoutCreateInfo set_2_info = {};
  set_2_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  set_2_info.pNext = nullptr;

  set_2_info.bindingCount = 1;
  set_2_info.flags = 0;
  set_2_info.pBindings = &object_buffer_binding;

  vkCreateDescriptorSetLayout(device_, &set_2_info, nullptr,
                              &object_set_layout_);

  VkDescriptorSetLayoutBinding texture_buffer_binding =
      vkinit::DescriptorSetLayoutBinding(
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          VK_SHADER_STAGE_FRAGMENT_BIT, 0);

  VkDescriptorSetLayoutCreateInfo set_3_info = {};
  set_3_info.sType = set_3_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  set_3_info.pNext = nullptr;

  set_3_info.bindingCount = 1;
  set_3_info.flags = 0;
  set_3_info.pBindings = &texture_buffer_binding;

  vkCreateDescriptorSetLayout(device_, &set_3_info, nullptr,
                              &single_texture_set_layout_);

  const size_t scene_parameter_buffer_size =
      kVkFrameOverlap * PadUniformBufferSize(sizeof(GpuSceneData));
  scene_parameter_buffer_ = CreateBuffer(scene_parameter_buffer_size,
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VMA_MEMORY_USAGE_CPU_TO_GPU);
  deletion_queue_.Push([&]() {
    vmaDestroyBuffer(allocator_, scene_parameter_buffer_.buffer,
                     scene_parameter_buffer_.allocation);
  });

  for (int i = 0; i < kVkFrameOverlap; i++) {
    constexpr int kMaxObjects = 10'000;
    frames_[i].object_buffer = CreateBuffer(sizeof(GpuObjectData) * kMaxObjects,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            VMA_MEMORY_USAGE_CPU_TO_GPU);

    frames_[i].camera_buffer =
        CreateBuffer(sizeof(GpuCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Allocate one descriptor set for each frame.
    VkDescriptorSetAllocateInfo allocation_info = {};
    allocation_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocation_info.pNext = nullptr;

    allocation_info.descriptorPool = descriptor_pool_;
    allocation_info.descriptorSetCount = 1;
    allocation_info.pSetLayouts = &global_set_layout_;

    vkAllocateDescriptorSets(device_, &allocation_info,
                             &frames_[i].global_descriptor);

    // Allocate the descriptor set that will point to object buffer.
    VkDescriptorSetAllocateInfo object_set_allocation_info = {};
    object_set_allocation_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    object_set_allocation_info.pNext = nullptr;

    object_set_allocation_info.descriptorPool = descriptor_pool_;
    object_set_allocation_info.descriptorSetCount = 1;
    object_set_allocation_info.pSetLayouts = &object_set_layout_;

    vkAllocateDescriptorSets(device_, &object_set_allocation_info,
                             &frames_[i].object_descriptor);

    // Make the descriptor set point to our camera buffer.
    VkDescriptorBufferInfo camera_buffer_info = {};
    camera_buffer_info.buffer = frames_[i].camera_buffer.buffer;
    camera_buffer_info.offset = 0;
    camera_buffer_info.range = sizeof(GpuCameraData);

    VkDescriptorBufferInfo scene_buffer_info = {};
    scene_buffer_info.buffer = scene_parameter_buffer_.buffer;
    // Note that we don't need an offset because this is dynamic.
    scene_buffer_info.offset = 0;
    scene_buffer_info.range = sizeof(GpuSceneData);

    VkDescriptorBufferInfo object_buffer_info = {};
    object_buffer_info.buffer = frames_[i].object_buffer.buffer;
    object_buffer_info.offset = 0;
    object_buffer_info.range = sizeof(GpuObjectData) * kMaxObjects;

    // Write into binding 0 of the global descriptor.
    VkWriteDescriptorSet camera_write = vkinit::WriteDescriptorBuffer(
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, frames_[i].global_descriptor,
        &camera_buffer_info, 0);

    VkWriteDescriptorSet scene_write = vkinit::WriteDescriptorBuffer(
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, frames_[i].global_descriptor,
        &scene_buffer_info, 1);

    VkWriteDescriptorSet object_write = vkinit::WriteDescriptorBuffer(
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frames_[i].object_descriptor,
        &object_buffer_info, 0);

    VkWriteDescriptorSet set_writes[] = {camera_write, scene_write,
                                         object_write};

    vkUpdateDescriptorSets(device_, 3, set_writes, 0, nullptr);

    deletion_queue_.Push([&, i]() {
      vmaDestroyBuffer(allocator_, frames_[i].camera_buffer.buffer,
                       frames_[i].camera_buffer.allocation);
      vmaDestroyBuffer(allocator_, frames_[i].object_buffer.buffer,
                       frames_[i].object_buffer.allocation);
    });
  }

  deletion_queue_.Push([&]() {
    vkDestroyDescriptorSetLayout(device_, global_set_layout_, nullptr);
    vkDestroyDescriptorSetLayout(device_, object_set_layout_, nullptr);
    vkDestroyDescriptorSetLayout(device_, single_texture_set_layout_, nullptr);
    vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
  });
}

void VulkanEngine::InitPipelines() {
  PipelineBuilder builder;
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

  builder.depth_stencil =
      vkinit::DepthStencilCreateInfo(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);

  VertexInputDescription vertex_description = Vertex::GetInputDescription();

  builder.vertex_input_info.pVertexAttributeDescriptions =
      vertex_description.attributes.data();
  builder.vertex_input_info.vertexAttributeDescriptionCount =
      vertex_description.attributes.size();

  builder.vertex_input_info.pVertexBindingDescriptions =
      vertex_description.bindings.data();
  builder.vertex_input_info.vertexBindingDescriptionCount =
      vertex_description.bindings.size();

  auto mesh_vert_shader =
      LoadShaderModule("../../shaders/triangle_mesh.vert.spv");
  if (mesh_vert_shader.has_value()) {
    std::cout << "Mesh triangle vertex shader successfully loaded.\n";
  } else {
    std::cerr << "Error when building mesh triangle vertex shader.\n";
  }

  auto default_lit_frag_shader =
      LoadShaderModule("../../shaders/default_lit.frag.spv");
  if (default_lit_frag_shader.has_value()) {
    std::cout << "Default lit fragment shader successfully loaded.\n";
  } else {
    std::cerr << "Error when building default lit fragment shader.\n";
  }

  builder.shader_stages.push_back(vkinit::PipelineShaderStageCreateInfo(
      VK_SHADER_STAGE_VERTEX_BIT, mesh_vert_shader.value()));
  builder.shader_stages.push_back(vkinit::PipelineShaderStageCreateInfo(
      VK_SHADER_STAGE_FRAGMENT_BIT, default_lit_frag_shader.value()));

  VkPushConstantRange push_constant;
  push_constant.offset = 0;
  push_constant.size = sizeof(MeshPushConstants);
  push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkPipelineLayoutCreateInfo mesh_pipeline_layout_info =
      vkinit::PipelineLayoutCreateInfo();
  mesh_pipeline_layout_info.pushConstantRangeCount = 1;
  mesh_pipeline_layout_info.pPushConstantRanges = &push_constant;

  VkDescriptorSetLayout set_layouts[] = {global_set_layout_,
                                         object_set_layout_};
  mesh_pipeline_layout_info.setLayoutCount = 2;
  mesh_pipeline_layout_info.pSetLayouts = set_layouts;

  VkPipelineLayout mesh_pipeline_layout;

  VK_CHECK(vkCreatePipelineLayout(device_, &mesh_pipeline_layout_info, nullptr,
                                  &mesh_pipeline_layout));

  builder.pipeline_layout = mesh_pipeline_layout;

  VkPipeline mesh_pipeline;
  mesh_pipeline = builder.BuildPipeline(device_, renderpass_);

  CreateMaterial(mesh_pipeline, mesh_pipeline_layout, "defaultmesh");

  // Create a pipeline layout for the textured mesh, which has 3 descriptor
  // sets.
  VkPipelineLayoutCreateInfo texture_pipeline_layout_info =
      mesh_pipeline_layout_info;

  VkDescriptorSetLayout textured_set_layouts[] = {
      global_set_layout_, object_set_layout_, single_texture_set_layout_};

  texture_pipeline_layout_info.setLayoutCount = 3;
  texture_pipeline_layout_info.pSetLayouts = textured_set_layouts;

  VkPipelineLayout textured_pipe_layout = {};
  VK_CHECK(vkCreatePipelineLayout(device_, &texture_pipeline_layout_info,
                                  nullptr, &textured_pipe_layout));

  auto textured_lit_frag_shader =
      LoadShaderModule("../../shaders/textured_lit.frag.spv");
  if (textured_lit_frag_shader.has_value()) {
    std::cout << "Textured lit fragment shader successfully loaded.\n";
  } else {
    std::cerr << "Error when building textured lit fragment shader.\n";
  }

  builder.shader_stages.clear();
  builder.shader_stages.push_back(vkinit::PipelineShaderStageCreateInfo(
      VK_SHADER_STAGE_VERTEX_BIT, mesh_vert_shader.value()));
  builder.shader_stages.push_back(vkinit::PipelineShaderStageCreateInfo(
      VK_SHADER_STAGE_FRAGMENT_BIT, textured_lit_frag_shader.value()));

  builder.pipeline_layout = textured_pipe_layout;

  VkPipeline textured_mesh_pipeline =
      builder.BuildPipeline(device_, renderpass_);

  CreateMaterial(textured_mesh_pipeline, textured_pipe_layout, "texturedmesh");

  vkDestroyShaderModule(device_, mesh_vert_shader.value(), nullptr);
  vkDestroyShaderModule(device_, default_lit_frag_shader.value(), nullptr);
  vkDestroyShaderModule(device_, textured_lit_frag_shader.value(), nullptr);

  deletion_queue_.Push([=]() {
    Material* material = GetMaterial("defaultmesh");
    if (!material) {
      return;
    }
    vkDestroyPipelineLayout(device_, material->pipeline_layout, nullptr);
    vkDestroyPipeline(device_, material->pipeline, nullptr);

    material = GetMaterial("texturedmesh");
    if (!material) {
      return;
    }
    vkDestroyPipelineLayout(device_, material->pipeline_layout, nullptr);
    vkDestroyPipeline(device_, material->pipeline, nullptr);
  });
}

void VulkanEngine::InitScene() {
  InitMonkeyScene();
  InitEmpireScene();
}

void VulkanEngine::InitMonkeyScene() {
  RenderObject monkey;
  monkey.mesh = GetMesh("monkey");
  monkey.material = GetMaterial("defaultmesh");
  monkey.transform = glm::mat4(1.f);
  renderables_.push_back(monkey);

  for (int x = -20; x <= 20; x++) {
    for (int y = -20; y <= 20; y++) {
      RenderObject triangle;
      triangle.mesh = GetMesh("triangle");
      triangle.material = GetMaterial("defaultmesh");

      glm::mat4 translation =
          glm::translate(glm::mat4(1.f), glm::vec3(x, 0, y));
      glm::mat4 scale = glm::scale(glm::mat4(1.f), glm::vec3(.2f, .2f, .2f));
      triangle.transform = translation * scale;

      renderables_.push_back(triangle);
    }
  }
}

void VulkanEngine::InitEmpireScene() {
  // Create a sampler for the texture.
  VkSamplerCreateInfo sampler_info =
      vkinit::SamplerCreateInfo(VK_FILTER_NEAREST);

  VkSampler blocky_sampler;
  vkCreateSampler(device_, &sampler_info, nullptr, &blocky_sampler);
  deletion_queue_.Push(
      [=]() { vkDestroySampler(device_, blocky_sampler, nullptr); });

  Material* textured_material = GetMaterial("texturedmesh");

  // Allocate the descriptor set for single-texture to use on the material.
  VkDescriptorSetAllocateInfo allocate_info = {};
  allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocate_info.pNext = nullptr;

  allocate_info.descriptorPool = descriptor_pool_;
  allocate_info.descriptorSetCount = 1;
  allocate_info.pSetLayouts = &single_texture_set_layout_;

  vkAllocateDescriptorSets(device_, &allocate_info,
                           &textured_material->texture_set);

  VkDescriptorImageInfo image_buffer_info = {};
  image_buffer_info.sampler = blocky_sampler;
  image_buffer_info.imageView = textures_["empire_diffuse"].image_view;
  image_buffer_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  VkWriteDescriptorSet texture1 = vkinit::WriteDescriptorImage(
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, textured_material->texture_set,
      &image_buffer_info, 0);

  vkUpdateDescriptorSets(device_, 1, &texture1, 0, nullptr);

  RenderObject map;
  map.mesh = GetMesh("empire");
  map.material = GetMaterial("texturedmesh");
  map.transform = glm::translate(glm::mat4(1), glm::vec3(5, -10, 0));

  renderables_.push_back(map);
}

void VulkanEngine::LoadTextures() {
  Texture lost_empire;
  LoadImageFromFile("../../assets/lost_empire-RGBA.png", lost_empire.image);

  VkImageViewCreateInfo image_info = vkinit::ImageViewCreateInfo(
      VK_FORMAT_R8G8B8A8_SRGB, lost_empire.image.image,
      VK_IMAGE_ASPECT_COLOR_BIT);

  vkCreateImageView(device_, &image_info, nullptr, &lost_empire.image_view);

  deletion_queue_.Push(
      [=]() { vkDestroyImageView(device_, lost_empire.image_view, nullptr); });

  textures_["empire_diffuse"] = lost_empire;
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

void VulkanEngine::LoadMeshes() {
  Mesh triangle_mesh;
  triangle_mesh.vertices.resize(3);

  // Vertex positions.
  triangle_mesh.vertices[0].position = {1.f, 1.f, 0.f};
  triangle_mesh.vertices[1].position = {-1.f, 1.f, 0.f};
  triangle_mesh.vertices[2].position = {0.f, -1.f, 0.f};

  // Vertex colors (all green).
  triangle_mesh.vertices[0].color = {0.f, 1.f, 0.f};
  triangle_mesh.vertices[1].color = {0.f, 1.f, 0.f};
  triangle_mesh.vertices[2].color = {0.f, 1.f, 0.f};

  // (We don't care about vertex normals).

  UploadMesh(triangle_mesh);

  meshes_["triangle"] = triangle_mesh;

  // Load monkey mesh.
  Mesh monkey_mesh;
  monkey_mesh.LoadFromObj("../../assets/monkey_smooth.obj");
  UploadMesh(monkey_mesh);

  meshes_["monkey"] = monkey_mesh;

  Mesh lost_empire;
  lost_empire.LoadFromObj("../../assets/lost_empire.obj");
  UploadMesh(lost_empire);
  meshes_["empire"] = lost_empire;
}

void VulkanEngine::UploadMesh(Mesh& mesh) {
  const size_t buffer_size = mesh.vertices.size() * sizeof(Vertex);

  // Allocate the staging buffer. This is only used to transfer to memory which
  // can then be copied to GPU-only memory.
  AllocatedBuffer staging_buffer = CreateBuffer(
      buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

  // Copy vertex data.
  void* data;
  vmaMapMemory(allocator_, staging_buffer.allocation, &data);
  memcpy(data, mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex));
  vmaUnmapMemory(allocator_, staging_buffer.allocation);

  // Allocate the vertex buffer.
  mesh.buffer = CreateBuffer(
      buffer_size,
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VMA_MEMORY_USAGE_GPU_ONLY);

  ImmediateSubmit([=](VkCommandBuffer cmd) {
    VkBufferCopy copy;
    copy.dstOffset = 0;
    copy.srcOffset = 0;
    copy.size = buffer_size;
    vkCmdCopyBuffer(cmd, staging_buffer.buffer, mesh.buffer.buffer, 1, &copy);
  });

  deletion_queue_.Push([=]() {
    vmaDestroyBuffer(allocator_, mesh.buffer.buffer, mesh.buffer.allocation);
  });

  // Since we submitted the command immediately, we don't need to wait to
  // destroy the staging buffer.
  vmaDestroyBuffer(allocator_, staging_buffer.buffer,
                   staging_buffer.allocation);
}

bool VulkanEngine::LoadImageFromFile(std::string filename,
                                     AllocatedImage& output) {
  int texture_width;
  int texture_height;
  int texture_channels;

  stbi_uc* pixels = stbi_load(filename.c_str(), &texture_width, &texture_height,
                              &texture_channels, STBI_rgb_alpha);

  if (!pixels) {
    return false;
  }

  // 4 bytes per pixel (R8G8B8A8).
  VkDeviceSize image_size = texture_width * texture_height * 4;

  // The format R8G8B8A8 matches with the pixels loaded from stb_image.
  VkFormat image_format = VK_FORMAT_R8G8B8A8_SRGB;

  AllocatedBuffer staging_buffer = CreateBuffer(
      image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);

  deletion_queue_.Push([=]() {
    vmaDestroyBuffer(allocator_, staging_buffer.buffer,
                     staging_buffer.allocation);
  });

  void* data;
  vmaMapMemory(allocator_, staging_buffer.allocation, &data);

  void* pixel_ptr = pixels;
  memcpy(data, pixel_ptr, static_cast<size_t>(image_size));

  vmaUnmapMemory(allocator_, staging_buffer.allocation);

  stbi_image_free(pixels);

  VkExtent3D image_extent;
  image_extent.width = static_cast<uint32_t>(texture_width);
  image_extent.height = static_cast<uint32_t>(texture_height);
  image_extent.depth = 1;

  VkImageCreateInfo image_info = vkinit::ImageCreateInfo(
      image_format,
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      image_extent);

  AllocatedImage image;

  VmaAllocationCreateInfo image_allocation_info = {};
  image_allocation_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

  // Allocate and create the image.
  vmaCreateImage(allocator_, &image_info, &image_allocation_info, &image.image,
                 &image.allocation, nullptr);

  ImmediateSubmit([&](VkCommandBuffer cmd) {
    // Note: We can't just copy the data from the buffer into the image
    // directly. The image is not initialized in any specific layout, so we need
    // to do a layout transition to put it in linear layout which is the best
    // layout for copying data from a buffer to a texture.

    // What part of the image we will transform (all of it).
    VkImageSubresourceRange range;
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = 0;
    range.levelCount = 1;
    range.baseArrayLayer = 0;
    range.layerCount = 1;

    VkImageMemoryBarrier image_barrier_to_transfer = {};
    image_barrier_to_transfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_barrier_to_transfer.pNext = nullptr;

    image_barrier_to_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_barrier_to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    image_barrier_to_transfer.image = image.image;
    image_barrier_to_transfer.subresourceRange = range;

    image_barrier_to_transfer.srcAccessMask = 0;
    image_barrier_to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    // Barrier the image into the transfer-receive layout.
    // Specify the source pipeline stage (TOP_OF_PIPE) and destination pipeline
    // stage (TRANSFER). See:
    // https://gpuopen.com/learn/vulkan-barriers-explained/
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &image_barrier_to_transfer);

    VkBufferImageCopy copy_region = {};
    copy_region.bufferOffset = 0;
    copy_region.bufferRowLength = 0;
    copy_region.bufferImageHeight = 0;

    copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.imageSubresource.mipLevel = 0;
    copy_region.imageSubresource.baseArrayLayer = 0;
    copy_region.imageSubresource.layerCount = 1;
    copy_region.imageExtent = image_extent;

    // Copy the buffer into the image.
    vkCmdCopyBufferToImage(cmd, staging_buffer.buffer, image.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &copy_region);

    // The image now has the correct pixel data, so we can change its layout one
    // more time to make it into a shader readable layout.
    VkImageMemoryBarrier image_barrier_to_readable = {};
    image_barrier_to_readable.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    image_barrier_to_readable.pNext = nullptr;

    image_barrier_to_readable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image_barrier_to_readable.newLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    image_barrier_to_readable.image = image.image;
    image_barrier_to_readable.subresourceRange = range;

    image_barrier_to_readable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    image_barrier_to_readable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &image_barrier_to_readable);
  });

  deletion_queue_.Push(
      [=]() { vmaDestroyImage(allocator_, image.image, image.allocation); });

  output = image;

  std::cout << "Texture Loaded: " << filename << std::endl;

  return true;
}

void VulkanEngine::ImmediateSubmit(
    std::function<void(VkCommandBuffer cmd)>&& function) {
  VkCommandBuffer& cmd = upload_context_.command_buffer;

  // Begin command buffer recording. We will use this command buffer exactly
  // once before resetting, so we tell Vulkan that.
  VkCommandBufferBeginInfo command_begin_info = vkinit::CommandBufferBeginInfo(
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

  VK_CHECK(vkBeginCommandBuffer(cmd, &command_begin_info));

  // Execute the function.
  function(cmd);

  VK_CHECK(vkEndCommandBuffer(cmd));

  VkSubmitInfo submit = vkinit::SubmitInfo(&cmd);

  // Submit the command buffer to the queue and execute it.
  // |upload_context_.upload_fence| will now block until the graphic commands
  // finish execution.
  VK_CHECK(
      vkQueueSubmit(graphics_queue_, 1, &submit, upload_context_.upload_fence));

  vkWaitForFences(device_, 1, &upload_context_.upload_fence, true,
                  kUploadSyncWaitTimeoutNs);
  vkResetFences(device_, 1, &upload_context_.upload_fence);

  // Reset the command buffers inside the command pool.
  vkResetCommandPool(device_, upload_context_.command_pool, 0);
}

AllocatedBuffer VulkanEngine::CreateBuffer(size_t allocation_size,
                                           VkBufferUsageFlags usage_flags,
                                           VmaMemoryUsage memory_usage) {
  // Allocate the vertex buffer.
  VkBufferCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.pNext = nullptr;

  info.size = allocation_size;
  info.usage = usage_flags;

  VmaAllocationCreateInfo vma_allocation_info = {};
  vma_allocation_info.usage = memory_usage;

  AllocatedBuffer buffer;

  VK_CHECK(vmaCreateBuffer(allocator_, &info, &vma_allocation_info,
                           &buffer.buffer, &buffer.allocation, nullptr));

  return buffer;
}

VulkanEngine::Material* VulkanEngine::CreateMaterial(VkPipeline pipeline,
                                                     VkPipelineLayout layout,
                                                     const std::string& name) {
  Material material;
  material.pipeline = pipeline;
  material.pipeline_layout = layout;
  materials_[name] = material;

  return &materials_[name];
}

VulkanEngine::FrameData& VulkanEngine::GetCurrentFrame() {
  return frames_[framenumber_ % kVkFrameOverlap];
}

VulkanEngine::Material* VulkanEngine::GetMaterial(const std::string& name) {
  auto it = materials_.find(name);
  if (it == materials_.end()) {
    return nullptr;
  }

  return &(*it).second;
}

Mesh* VulkanEngine::GetMesh(const std::string& name) {
  auto it = meshes_.find(name);
  if (it == meshes_.end()) {
    return nullptr;
  }

  return &(*it).second;
}

void VulkanEngine::DrawObjects(VkCommandBuffer cmd, RenderObject* first,
                               int count) {
  // Make a mvp matrix.
  glm::vec3 camera_position = {0.f, -6.f, -10.f};
  glm::mat4 view = glm::translate(glm::mat4(1.f), camera_position);

  glm::mat4 projection =
      glm::perspective(glm::radians(70.f), 1700.f / 900.f, 0.1f, 200.f);
  projection[1][1] *= -1;

  GpuCameraData camera_data;
  camera_data.projection = projection;
  camera_data.view = view;
  camera_data.view_projection = projection * view;

  FrameData& frame = GetCurrentFrame();

  // Camera data
  void* data;
  vmaMapMemory(allocator_, frame.camera_buffer.allocation, &data);
  memcpy(data, &camera_data, sizeof(GpuCameraData));
  vmaUnmapMemory(allocator_, frame.camera_buffer.allocation);

  float framed = framenumber_ / 120.f;
  scene_parameters_.ambient_color = {sin(framed), 0, cos(framed), 1};

  // Scene Data
  char* scene_data;
  vmaMapMemory(allocator_, scene_parameter_buffer_.allocation,
               (void**)&scene_data);

  int frame_index = framenumber_ % kVkFrameOverlap;
  scene_data += PadUniformBufferSize(sizeof(GpuSceneData)) * frame_index;
  memcpy(scene_data, &scene_parameters_, sizeof(GpuSceneData));

  vmaUnmapMemory(allocator_, scene_parameter_buffer_.allocation);

  void* object_data;
  vmaMapMemory(allocator_, frame.object_buffer.allocation, &object_data);

  GpuObjectData* object_ssbo = (GpuObjectData*)object_data;
  for (int i = 0; i < count; i++) {
    RenderObject& object = first[i];
    object_ssbo[i].model_matrix = object.transform;
  }

  vmaUnmapMemory(allocator_, frame.object_buffer.allocation);

  Mesh* last_mesh = nullptr;
  Material* last_material = nullptr;

  for (int i = 0; i < count; i++) {
    RenderObject& object = first[i];
    if (object.material != last_material) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        object.material->pipeline);
      last_material = object.material;

      uint32_t uniform_offset =
          PadUniformBufferSize(sizeof(GpuSceneData) * frame_index);

      // Bind the descriptor set when changing the pipeline.
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              object.material->pipeline_layout, 0, 1,
                              &frame.global_descriptor, 1, &uniform_offset);

      // Bind the object data descriptor.
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              object.material->pipeline_layout, 1, 1,
                              &frame.object_descriptor, 0, nullptr);

      // Bind the texture descriptor.
      if (object.material->texture_set) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                object.material->pipeline_layout, 2, 1,
                                &object.material->texture_set, 0, nullptr);
      }
    }

    glm::mat4 mvp = projection * view * object.transform;

    MeshPushConstants constants;
    constants.matrix = object.transform;

    vkCmdPushConstants(cmd, object.material->pipeline_layout,
                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants),
                       &constants);

    if (object.mesh != last_mesh) {
      VkDeviceSize offset = 0;
      vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->buffer.buffer, &offset);
      last_mesh = object.mesh;
    }

    // Use the loop index to send the instance index to the shader.
    vkCmdDraw(cmd, object.mesh->vertices.size(), 1, 0, i);
  }
}

size_t VulkanEngine::PadUniformBufferSize(size_t original_size) {
  // Calculate the required alignment based on the minimum device offset
  // alignment.
  size_t minimum_alignment =
      gpu_properties_.limits.minUniformBufferOffsetAlignment;
  size_t aligned_size = original_size;
  if (minimum_alignment > 0) {
    aligned_size =
        (aligned_size + minimum_alignment - 1) & ~(minimum_alignment - 1);
  }

  return aligned_size;
}

void VulkanEngine::Cleanup() {
  if (is_initialized_) {
    // Wait for VkCommandBuffer to finish before destroying render pass.
    for (int i = 0; i < kVkFrameOverlap; i++) {
      VK_CHECK(vkWaitForFences(device_, 1, &frames_[i].render_fence, true,
                               kSyncWaitTimeoutNs));
    }

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
  FrameData& frame = GetCurrentFrame();
  VK_CHECK(vkWaitForFences(device_, 1, &frame.render_fence, true,
                           kSyncWaitTimeoutNs));
  VK_CHECK(vkResetFences(device_, 1, &frame.render_fence));

  // Request image from the swapchain, one second timeout.
  uint32_t swapchain_image_index;
  VK_CHECK(vkAcquireNextImageKHR(device_, swapchain_, kSyncWaitTimeoutNs,
                                 frame.present_semaphore, nullptr,
                                 &swapchain_image_index));

  // Now that we're sure that the commands have finished executing, we can
  // safely reset the command buffer to begin recording again.
  VK_CHECK(vkResetCommandBuffer(frame.command_buffer, 0));

  // Begin command buffer recording. We will use this command buffer exactly
  // once, so we want to let Vulkan know that.
  VkCommandBufferBeginInfo command_begin_info = vkinit::CommandBufferBeginInfo(
      VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

  VK_CHECK(vkBeginCommandBuffer(frame.command_buffer, &command_begin_info));

  // DRAW

  VkClearValue color_clear_value;
  float flash = abs(sin(framenumber_ / 120.f));
  color_clear_value.color = {{0.0f, 0.0f, flash, 1.0f}};

  VkClearValue depth_clear_value;
  depth_clear_value.depthStencil.depth = 1.f;

  // Main renderpass.
  VkRenderPassBeginInfo renderpass_begin_info = {};
  renderpass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderpass_begin_info.pNext = nullptr;

  renderpass_begin_info.renderPass = renderpass_;
  renderpass_begin_info.renderArea.offset.x = 0;
  renderpass_begin_info.renderArea.offset.y = 0;
  renderpass_begin_info.renderArea.extent = window_extent_;
  renderpass_begin_info.framebuffer = framebuffers_[swapchain_image_index];

  VkClearValue clear_values[2] = {color_clear_value, depth_clear_value};
  renderpass_begin_info.clearValueCount = 2;
  renderpass_begin_info.pClearValues = &clear_values[0];

  vkCmdBeginRenderPass(frame.command_buffer, &renderpass_begin_info,
                       VK_SUBPASS_CONTENTS_INLINE);

  DrawObjects(frame.command_buffer, renderables_.data(), renderables_.size());

  // Finalize the render pass and command buffer.
  vkCmdEndRenderPass(frame.command_buffer);
  VK_CHECK(vkEndCommandBuffer(frame.command_buffer));

  // Prepare the submission to the queue.
  // We want to wait on the |present_semaphore_|, as that semaphore is signaled
  // when the swapchain is ready. We will use the |render_semaphore_| when
  // rendering has finish.
  VkPipelineStageFlags wait_stage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo submit_info =
      vkinit::SubmitInfo(&frame.command_buffer, &wait_stage,
                         &frame.present_semaphore, &frame.render_semaphore);
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.pNext = nullptr;

  submit_info.pWaitDstStageMask = &wait_stage;

  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = &frame.present_semaphore;

  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = &frame.render_semaphore;

  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &frame.command_buffer;

  // Submit command buffer to the queue and execute it.
  // |render_fence_| will now block until the graphic command finishes.
  VK_CHECK(vkQueueSubmit(graphics_queue_, 1, &submit_info, frame.render_fence));

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
  present_info.pWaitSemaphores = &frame.render_semaphore;

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
  pipeline_info.pDepthStencilState = &depth_stencil;

  VkPipeline pipeline;
  if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                nullptr, &pipeline) != VK_SUCCESS) {
    std::cerr << "Failed to create pipeline\n";
    return VK_NULL_HANDLE;
  }

  return pipeline;
}
