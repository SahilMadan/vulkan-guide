#version 450

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec3 vColor;

layout (location = 0) out vec3 outColor;

layout(set = 0, binding = 0) uniform CameraBuffer {
    mat4 view;
    mat4 projection;
    mat4 view_projection;
} camera_data;

layout( push_constant ) uniform constants {
    vec4 data;
    mat4 matrix;
} push_constants;

void main() {
    mat4 transform_matrix = (camera_data.view_projection * push_constants.matrix);
    gl_Position = transform_matrix * vec4(vPosition, 1.f);
    outColor = vColor;
}