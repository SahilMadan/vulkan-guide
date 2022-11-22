#version 450

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 texCoord;

layout (set = 2, binding = 0) uniform sampler2D texture1;

layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 1) uniform SceneData {
    vec4 fog_color;
    vec4 fog_distance;
    vec4 ambient_color;
    vec4 sunlight_direction;
    vec4 sunlight_color;
} scene_data;

void main() {
    vec3 color = texture(texture1, texCoord).xyz;
    outFragColor = vec4(color, 1.f);
}