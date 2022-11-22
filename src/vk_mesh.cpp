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

  // UV stored at location 3.
  VkVertexInputAttributeDescription uv = {};
  uv.binding = 0;
  uv.location = 3;
  uv.format = VK_FORMAT_R32G32_SFLOAT;
  uv.offset = offsetof(Vertex, uv);

  description.attributes.push_back(position);
  description.attributes.push_back(normal);
  description.attributes.push_back(color);
  description.attributes.push_back(uv);

  return description;
}

bool Mesh::LoadFromObj(const char* filename) {
  // Attrib will contain the vertex arrays of the file.
  tinyobj::attrib_t attrib;
  // Shapes will contain the info for each separate object in the file.
  std::vector<tinyobj::shape_t> shapes;
  // Materials will contain the information about the materials of each shape.
  // TODO: Currently unused.
  std::vector<tinyobj::material_t> materials;

  std::string error;
  std::string warning;

  tinyobj::LoadObj(&attrib, &shapes, &materials, &warning, &error, filename,
                   /*readMaterialFunc=*/nullptr);

  if (!error.empty()) {
    std::cerr << "Error: " << error << std::endl;
    return false;
  }

  if (!warning.empty()) {
    std::cout << "Warning: " << warning << std::endl;
  }

  for (size_t s = 0; s < shapes.size(); s++) {
    // Loop over the faces (polygon).
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      // Hardcode loading to triangle.
      int fv = 3;

      // Loop over vertices in the faces.
      for (size_t v = 0; v < fv; v++) {
        tinyobj::index_t index = shapes[s].mesh.indices[index_offset + v];

        // Vertex position.
        tinyobj::real_t vx = attrib.vertices[3 * index.vertex_index + 0];
        tinyobj::real_t vy = attrib.vertices[3 * index.vertex_index + 1];
        tinyobj::real_t vz = attrib.vertices[3 * index.vertex_index + 2];

        // Vertex normal.
        tinyobj::real_t nx = attrib.normals[3 * index.normal_index + 0];
        tinyobj::real_t ny = attrib.normals[3 * index.normal_index + 1];
        tinyobj::real_t nz = attrib.normals[3 * index.normal_index + 2];

        // Vertex uv.
        tinyobj::real_t ux = attrib.texcoords[2 * index.texcoord_index + 0];
        tinyobj::real_t uy = attrib.texcoords[2 * index.texcoord_index + 1];

        Vertex vertex;
        vertex.position.x = vx;
        vertex.position.y = vy;
        vertex.position.z = vz;

        vertex.normal.x = nx;
        vertex.normal.y = ny;
        vertex.normal.z = nz;

        vertex.uv.x = ux;
        vertex.uv.y = 1 - uy;

        // Set the vertex color as the vertex normal.
        vertex.color = vertex.normal;

        vertices.push_back(vertex);
      }
      index_offset += fv;
    }
  }
  return false;
}
