
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

  InitVulkan();
  InitSwapchain();
  InitCommands();
  InitDefaultRenderpass();
  InitFramebuffers();
  InitSyncStructs();
  InitDescriptors();
  InitPipelines();

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
  vkb::Device vkb_device = device_builder.build().value();

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
}

void VulkanEngine::InitDescriptors() {
  // Create a descriptor pool that will hold 10 uniform buffers.
  std::vector<VkDescriptorPoolSize> sizes = {
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10}};

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

  VkDescriptorSetLayoutBinding camera_buffer_binding = {};
  camera_buffer_binding.binding = 0;
  camera_buffer_binding.descriptorCount = 1;
  camera_buffer_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

  camera_buffer_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkDescriptorSetLayoutCreateInfo set_info = {};
  set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  set_info.pNext = nullptr;

  set_info.bindingCount = 1;
  set_info.flags = 0;
  set_info.pBindings = &camera_buffer_binding;

  vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &global_set_layout_);

  for (int i = 0; i < kVkFrameOverlap; i++) {
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

    // Make the descriptor set point to our camera buffer.
    VkDescriptorBufferInfo buffer_info = {};
    buffer_info.buffer = frames_[i].camera_buffer.buffer;
    buffer_info.offset = 0;
    buffer_info.range = sizeof(GpuCameraData);

    VkWriteDescriptorSet set_write = {};
    set_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    set_write.pNext = nullptr;

    // Write into binding number 0...
    set_write.dstBinding = 0;
    // ... of the global descriptor.
    set_write.dstSet = frames_[i].global_descriptor;

    set_write.descriptorCount = 1;
    set_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    set_write.pBufferInfo = &buffer_info;

    vkUpdateDescriptorSets(device_, 1, &set_write, 0, nullptr);
  }

  for (int i = 0; i < kVkFrameOverlap; i++) {
    deletion_queue_.Push([&, i]() {
      vmaDestroyBuffer(allocator_, frames_[i].camera_buffer.buffer,
                       frames_[i].camera_buffer.allocation);
    });
  }

  deletion_queue_.Push([&]() {
    vkDestroyDescriptorSetLayout(device_, global_set_layout_, nullptr);
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

  auto colored_triangle_frag_shader =
      LoadShaderModule("../../shaders/colored_triangle.frag.spv");
  if (colored_triangle_frag_shader.has_value()) {
    std::cout << "Colored triangle fragment shader successfully loaded.\n";
  } else {
    std::cerr << "Error when building colored triangle fragment shader.\n";
  }

  builder.shader_stages.push_back(vkinit::PipelineShaderStageCreateInfo(
      VK_SHADER_STAGE_VERTEX_BIT, mesh_vert_shader.value()));
  builder.shader_stages.push_back(vkinit::PipelineShaderStageCreateInfo(
      VK_SHADER_STAGE_FRAGMENT_BIT, colored_triangle_frag_shader.value()));

  VkPushConstantRange push_constant;
  push_constant.offset = 0;
  push_constant.size = sizeof(MeshPushConstants);
  push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkPipelineLayoutCreateInfo mesh_pipeline_layout_info =
      vkinit::PipelineLayoutCreateInfo();
  mesh_pipeline_layout_info.pushConstantRangeCount = 1;
  mesh_pipeline_layout_info.pPushConstantRanges = &push_constant;

  mesh_pipeline_layout_info.setLayoutCount = 1;
  mesh_pipeline_layout_info.pSetLayouts = &global_set_layout_;

  VkPipelineLayout mesh_pipeline_layout;

  VK_CHECK(vkCreatePipelineLayout(device_, &mesh_pipeline_layout_info, nullptr,
                                  &mesh_pipeline_layout));

  builder.pipeline_layout = mesh_pipeline_layout;

  VkPipeline mesh_pipeline;
  mesh_pipeline = builder.BuildPipeline(device_, renderpass_);

  vkDestroyShaderModule(device_, mesh_vert_shader.value(), nullptr);
  vkDestroyShaderModule(device_, colored_triangle_frag_shader.value(), nullptr);

  CreateMaterial(mesh_pipeline, mesh_pipeline_layout, "defaultmesh");

  deletion_queue_.Push([=]() {
    Material* material = GetMaterial("defaultmesh");
    if (!material) {
      return;
    }
    vkDestroyPipelineLayout(device_, material->pipeline_layout, nullptr);
    vkDestroyPipeline(device_, material->pipeline, nullptr);
  });
}

void VulkanEngine::InitScene() {
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
}

void VulkanEngine::UploadMesh(Mesh& mesh) {
  mesh.buffer = CreateBuffer(mesh.vertices.size() * sizeof(Vertex),
                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VMA_MEMORY_USAGE_CPU_TO_GPU);

  deletion_queue_.Push([=]() {
    vmaDestroyBuffer(allocator_, mesh.buffer.buffer, mesh.buffer.allocation);
  });

  // Copy vertex data.
  void* data;
  vmaMapMemory(allocator_, mesh.buffer.allocation, &data);
  memcpy(data, mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex));
  vmaUnmapMemory(allocator_, mesh.buffer.allocation);
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

  void* data;
  vmaMapMemory(allocator_, frame.camera_buffer.allocation, &data);
  memcpy(data, &camera_data, sizeof(GpuCameraData));
  vmaUnmapMemory(allocator_, frame.camera_buffer.allocation);

  Mesh* last_mesh = nullptr;
  Material* last_material = nullptr;

  for (int i = 0; i < count; i++) {
    RenderObject& object = first[i];
    if (object.material != last_material) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        object.material->pipeline);
      last_material = object.material;

      // Bind the descriptor set when changing the pipeline.
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              object.material->pipeline_layout, 0, 1,
                              &frame.global_descriptor, 0, nullptr);
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

    vkCmdDraw(cmd, object.mesh->vertices.size(), 1, 0, 0);
  }
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
  VkCommandBufferBeginInfo command_begin_info = {};
  command_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  command_begin_info.pNext = nullptr;

  command_begin_info.pInheritanceInfo = nullptr;
  command_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

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
  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.pNext = nullptr;

  VkPipelineStageFlags wait_stage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

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
