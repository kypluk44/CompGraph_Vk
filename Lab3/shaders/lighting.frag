#version 450
layout(location=0) in vec3 worldPosition;
layout(location=1) in vec3 worldNormal;
layout(location=0) out vec4 outColor;
layout(location=2) in vec2 uv;
layout(set=1,binding=1) uniform sampler2D albedoMap;
layout(set=1,binding=2) uniform sampler2D specularMap;
layout(set=1,binding=3) uniform sampler2D emissiveMap;
layout(push_constant) uniform Sampling { vec4 effect; } sampling;
vec3 surfaceAlbedo;
vec3 surfaceSpecular;
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

struct PointLight { vec4 position; vec4 colorIntensity; };
struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 colorIntensity;
    vec4 coneCosines;
};
// Массивы не имеют фиксированной длины; CPU передаёт фактическое число источников.
layout(set=0,binding=1,std430) readonly buffer PointLights { PointLight values[]; } points;
layout(set=0,binding=2,std430) readonly buffer SpotLights { SpotLight values[]; } spots;

vec3 safeNormalize(vec3 v) {
    return v * inversesqrt(max(dot(v,v),1e-8));
}
// Одна модель Блинн-Фонга применяется к каждому типу источника света.
vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance) {
    float diffuse = max(dot(N,L),0.0);
    vec3 H = safeNormalize(L+V);
    float specular = diffuse > 0.0
        ? pow(max(dot(N,H),0.0),model.specularShininess.w) : 0.0;
    return radiance * (surfaceAlbedo * diffuse + surfaceSpecular * specular);
}
void main() {
    vec2 st=uv;
    vec3 texel=texture(albedoMap,st).rgb;
    if(sampling.effect.x>0.5){
        float t=sampling.effect.y;
        st=uv+0.12*vec2(sin(t),cos(t*0.7));
        texel=mix(texture(albedoMap,st).rgb,texture(albedoMap,uv*2.0-vec2(t*0.06,0)).rgb,0.35);
        texel*=0.7+0.3*cos(t+vec3(0,2,4));
    }
    surfaceAlbedo=texel*model.albedo.rgb;
    surfaceSpecular=texture(specularMap,st).rgb*model.specularShininess.rgb;
    vec3 N = normalize(worldNormal);
    vec3 V = safeNormalize(scene.cameraPosition.xyz-worldPosition);
    vec3 color = scene.ambient.rgb * scene.ambient.w * surfaceAlbedo;
    color += blinnPhong(N,V,-scene.directionalDirection.xyz,
                       scene.directionalColor.rgb * scene.directionalColor.w);
    for (uint i=0u; i<scene.lightCounts.x; ++i) {
        vec3 delta = points.values[i].position.xyz-worldPosition;
        float distanceSquared = max(dot(delta,delta),1e-4);
        vec3 radiance = points.values[i].colorIntensity.rgb * points.values[i].colorIntensity.w / distanceSquared;
        color += blinnPhong(N,V,safeNormalize(delta),radiance);
    }
    for (uint i=0u; i<scene.lightCounts.y; ++i) {
        vec3 delta = spots.values[i].position.xyz-worldPosition;
        vec3 L = safeNormalize(delta);
        float distanceSquared = max(dot(delta,delta),1e-4);
        float cosine = dot(-L,spots.values[i].direction.xyz);
        float innerCos = spots.values[i].coneCosines.x;
        float outerCos = spots.values[i].coneCosines.y;
        // Между внутренним и внешним конусами плавно уменьшаем яркость до нуля.
        float cone = innerCos > outerCos+1e-6
            ? smoothstep(outerCos,innerCos,cosine) : step(outerCos,cosine);
        vec3 radiance = spots.values[i].colorIntensity.rgb * spots.values[i].colorIntensity.w * cone / distanceSquared;
        color += blinnPhong(N,V,L,radiance);
    }
    outColor = vec4(color+texture(emissiveMap,st).rgb,1.0);
}
