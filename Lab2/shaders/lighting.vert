#version 450
layout(location=0) in vec3 inPosition;
layout(location=1) in vec3 inNormal;
layout(location=0) out vec3 worldPosition;
layout(location=1) out vec3 worldNormal;
layout(set=0,binding=0,std140) uniform Scene {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 directionalDirection;
    vec4 directionalColor;
    uvec4 lightCounts;
} scene;
layout(set=1,binding=0,std140) uniform Model {
    mat4 transform;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specularShininess;
} model;

void main() {
    vec4 world = model.transform * vec4(inPosition,1.0);
    worldPosition = world.xyz;
    worldNormal = mat3(model.normalMatrix) * inNormal;
    gl_Position = scene.viewProjection * world;
}
