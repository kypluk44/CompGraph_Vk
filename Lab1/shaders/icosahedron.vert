#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(set = 0, binding = 0, std140) uniform GlobalUniforms {
	mat4 view;
	mat4 projection;
} globalUniforms;

layout(set = 1, binding = 0, std140) uniform ObjectUniforms {
	mat4 model;
	vec4 tint;
} objectUniforms;

layout(location = 0) out vec3 vertexColor;

void main() {
	gl_Position = globalUniforms.projection * globalUniforms.view *
	              objectUniforms.model * vec4(inPosition, 1.0);
	vertexColor = inColor * objectUniforms.tint.rgb;
}
