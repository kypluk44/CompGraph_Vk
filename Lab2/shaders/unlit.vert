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

layout(push_constant) uniform Draw { mat4 transform; vec4 color; } draw;
void main() {
    gl_Position = scene.viewProjection * draw.transform * vec4(inPosition,1.0);
    markerColor = draw.color;
}
