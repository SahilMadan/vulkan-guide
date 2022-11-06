#pragma once

#include <glm/vec3.hpp>
#include <vector>
#include <vk_types.hpp>

struct VertexInputDescription {
  std::vector<VkVertexInputBindingDescription> bindings;
  std::vector<VkVertexInputAttributeDescription> attributes;

  VkPipelineVertexInputStateCreateFlags flags = 0;
};

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec3 color;

  static VertexInputDescription GetInputDescription();
};

struct Mesh {
  std::vector<Vertex> vertices;
  AllocatedBuffer buffer;
};
