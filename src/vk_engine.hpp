// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_mem_alloc.h>

#include <deletion_queue.hpp>
#include <glm/glm.hpp>
#include <optional>
#include <string>
#include <vector>
#include <vk_mesh.hpp>
#include <vk_types.hpp>

class VulkanEngine {
 public:
  bool is_initialized_{false};
  int framenumber_{0};

  VkExtent2D window_extent_{1700, 900};

  struct SDL_Window* window_{nullptr};

  // initializes everything in the engine
  void Init();

  // shuts down the engine
  void Cleanup();

  // draw loop
  void Draw();

  // run main loop
  void Run();

 private:
  struct PipelineBuilder {
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;
    VkPipelineVertexInputStateCreateInfo vertex_input_info;
    VkPipelineInputAssemblyStateCreateInfo input_assembly;
    VkViewport viewport;
    VkRect2D scissor;
    VkPipelineRasterizationStateCreateInfo rasterizer;
    VkPipelineColorBlendAttachmentState color_blend_attachment;
    VkPipelineMultisampleStateCreateInfo multisampling;
    VkPipelineLayout pipeline_layout;
    VkPipelineDepthStencilStateCreateInfo depth_stencil;

    VkPipeline BuildPipeline(VkDevice device, VkRenderPass renderpass);
  };

  struct MeshPushConstants {
    glm::vec4 data;
    glm::mat4 matrix;
  };

  VkInstance instance_;
  VkDebugUtilsMessengerEXT debug_messenger_;
  VkPhysicalDevice gpu_;
  VkDevice device_;
  VkSurfaceKHR surface_;

  VkSwapchainKHR swapchain_;
  VkFormat swapchain_image_format_;
  std::vector<VkImage> swapchain_images_;
  std::vector<VkImageView> swapchain_image_views_;

  VkImageView depth_image_view_;
  AllocatedImage depth_image_;
  VkFormat depth_format_;

  VkQueue graphics_queue_;
  uint32_t graphics_queue_family_;
  VkCommandPool command_pool_;
  VkCommandBuffer command_buffer_;

  VkRenderPass renderpass_;
  std::vector<VkFramebuffer> framebuffers_;

  // Semaphores used for GPU <-> GPU sync.
  VkSemaphore present_semaphore_;
  VkSemaphore render_semaphore_;
  // Fence used for CPU <-> GPU sync.
  VkFence render_fence_;

  VkPipelineLayout triangle_pipeline_layout_;
  VkPipelineLayout mesh_pipeline_layout_;

  VkPipeline triangle_pipeline_;
  VkPipeline colored_triangle_pipeline_;
  VkPipeline mesh_pipeline_;

  VmaAllocator allocator_;

  Mesh triangle_mesh_;
  Mesh monkey_mesh_;

  int selected_shader_ = 0;

  DeletionQueue deletion_queue_;

  // Initialization Helpers.
  void InitVulkan();
  void InitSwapchain();
  void InitCommands();
  void InitDefaultRenderpass();
  void InitFramebuffers();
  void InitSyncStructs();
  void InitPipelines();

  void LoadMeshes();
  void UploadMesh(Mesh& mesh);

  std::optional<VkShaderModule> LoadShaderModule(const std::string& path);
};
