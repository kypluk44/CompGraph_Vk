#version 450
layout(location=0) in vec3 inPosition;

layout(set=1,binding=0,std140) uniform Model {
    mat4 transform;
} model;

layout(push_constant) uniform Light {
    mat4 viewProjection;
} light;

void main() {
    gl_Position = light.viewProjection * model.transform * vec4(inPosition, 1.0);
}
