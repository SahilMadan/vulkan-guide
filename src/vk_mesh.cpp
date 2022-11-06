#include <vk_mesh.hpp>

VertexInputDescription Vertex::GetInputDescription() {
  VertexInputDescription description;

  // We will have just 1 vertex buffer binding, with a per-vertex rate.
  VkVertexInputBindingDescription binding = {};
  binding.binding = 0;
  binding.stride = sizeof(Vertex);
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  description.bindings.push_back(binding);

  // Position stored at location 0.
  VkVertexInputAttributeDescription position = {};
  position.binding = 0;
  position.location = 0;
  position.format = VK_FORMAT_R32G32B32_SFLOAT;
  position.offset = offsetof(Vertex, position);

  // Normal stored at location 1.
  VkVertexInputAttributeDescription normal = {};
  normal.binding = 0;
  normal.location = 1;
  normal.format = VK_FORMAT_R32G32B32_SFLOAT;
  normal.offset = offsetof(Vertex, normal);

  // Color stored at location 2.
  VkVertexInputAttributeDescription color = {};
  color.binding = 0;
  color.location = 2;
  color.format = VK_FORMAT_R32G32B32_SFLOAT;
  color.offset = offsetof(Vertex, color);

  description.attributes.push_back(position);
  description.attributes.push_back(normal);
  description.attributes.push_back(color);

  return description;
}
