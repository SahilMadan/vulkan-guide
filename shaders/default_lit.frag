#version 450

layout (location = 0) in vec3 inColor;

layout (location = 0) out vec4 outFragColor;

layout(set = 0, binding = 1) uniform SceneData {
    vec4 fog_color;
    vec4 fog_distance;
    vec4 ambient_color;
    vec4 sunlight_direction;
    vec4 sunlight_color;
} scene_data;

void main() {
    outFragColor = vec4(inColor + scene_data.ambient_color.xyz, 1.f);
}