#version 450
layout(location=0) in vec3 inPosition;
layout(location=0) out vec4 markerColor;
layout(set=0,binding=0,std140) uniform Scene {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 directionalDirection;
    vec4 directionalColor;
    uvec4 lightCounts;
} scene;

layout(set=1,binding=0,std140) uniform Model { mat4 transform; mat4 normalMatrix; vec4 albedo; vec4 specularShininess; } model;
void main() {
    gl_Position = scene.viewProjection * model.transform * vec4(inPosition,1.0);
    markerColor = model.albedo;
}
