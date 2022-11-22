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

constexpr unsigned int kVkFrameOverlap = 2;

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

  struct GpuCameraData {
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 view_projection;
  };

  struct GpuSceneData {
    // w is for exponent
    glm::vec4 fog_color;
    // x for min, y for max, zw unused.
    glm::vec4 fog_distance;

    glm::vec4 ambient_color;
    // w is for sun power.
    glm::vec4 sunlight_direction;
    glm::vec4 sunlight_color;
  };

  struct GpuObjectData {
    glm::mat4 model_matrix;
  };

  struct FrameData {
    // Semaphores used for GPU <-> GPU sync.
    VkSemaphore present_semaphore;
    VkSemaphore render_semaphore;
    // Fence used for CPU <-> GPU sync.
    VkFence render_fence;

    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    AllocatedBuffer camera_buffer;
    VkDescriptorSet global_descriptor;

    AllocatedBuffer object_buffer;
    VkDescriptorSet object_descriptor;
  };

  struct MeshPushConstants {
    glm::vec4 data;
    glm::mat4 matrix;
  };

  struct Material {
    VkPipeline pipeline;
    VkPipelineLayout pipeline_layout;
  };

  struct RenderObject {
    Mesh* mesh;
    Material* material;
    glm::mat4 transform;
  };

  VkInstance instance_;
  VkDebugUtilsMessengerEXT debug_messenger_;
  VkPhysicalDevice gpu_;
  VkDevice device_;
  VkSurfaceKHR surface_;

  VkPhysicalDeviceProperties gpu_properties_;

  VkSwapchainKHR swapchain_;
  VkFormat swapchain_image_format_;
  std::vector<VkImage> swapchain_images_;
  std::vector<VkImageView> swapchain_image_views_;

  VkImageView depth_image_view_;
  AllocatedImage depth_image_;
  VkFormat depth_format_;

  VkQueue graphics_queue_;
  uint32_t graphics_queue_family_;

  FrameData frames_[kVkFrameOverlap];

  VkRenderPass renderpass_;
  std::vector<VkFramebuffer> framebuffers_;

  VmaAllocator allocator_;

  std::vector<RenderObject> renderables_;

  std::unordered_map<std::string, Material> materials_;
  std::unordered_map<std::string, Mesh> meshes_;

  VkDescriptorSetLayout global_set_layout_;
  VkDescriptorSetLayout object_set_layout_;
  VkDescriptorPool descriptor_pool_;

  GpuSceneData scene_parameters_;
  AllocatedBuffer scene_parameter_buffer_;

  DeletionQueue deletion_queue_;

  // Initialization Helpers.
  void InitVulkan();
  void InitSwapchain();
  void InitCommands();
  void InitDefaultRenderpass();
  void InitFramebuffers();
  void InitSyncStructs();
  void InitDescriptors();
  void InitPipelines();
  void InitScene();

  void LoadMeshes();
  void UploadMesh(Mesh& mesh);

  AllocatedBuffer CreateBuffer(size_t allocation_size,
                               VkBufferUsageFlags usage_flags,
                               VmaMemoryUsage memory_usage);

  Material* CreateMaterial(VkPipeline pipeline, VkPipelineLayout layout,
                           const std::string& name);

  FrameData& GetCurrentFrame();
  Material* GetMaterial(const std::string& name);
  Mesh* GetMesh(const std::string& name);

  void DrawObjects(VkCommandBuffer cmd, RenderObject* first, int count);

  std::optional<VkShaderModule> LoadShaderModule(const std::string& path);

  size_t PadUniformBufferSize(size_t original_size);
};
