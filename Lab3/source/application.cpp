#include "application.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace application {
namespace {

// Нормали вычисляются из геометрии; для каждой грани используются свои вершины.
struct Vertex { glm::vec3 position; glm::vec3 normal; glm::vec2 uv{}; };
struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};
struct Mesh { Buffer vertices; Buffer indices; uint32_t index_count = 0; };

// Порядок полей совпадает с блоками std140/std430 в GLSL.
struct SceneUniforms {
    glm::mat4 view_projection;
    glm::vec4 camera_position;
    glm::vec4 ambient;
    glm::vec4 directional_direction;
    glm::vec4 directional_color;
    glm::uvec4 light_counts;
};
struct ModelUniforms {
    glm::mat4 model;
    glm::mat4 normal_matrix;
    glm::vec4 albedo;
    glm::vec4 specular_shininess;
};
struct PointGpu { glm::vec4 position; glm::vec4 color_intensity; };
struct SpotGpu {
    glm::vec4 position;
    glm::vec4 direction;
    glm::vec4 color_intensity;
    glm::vec4 cone_cosines;
};
// Push constants передают флаг сэмплирования и время; свойства объектов лежат в dynamic UBO.
struct DrawPush { glm::vec4 effect; };
struct Texture { VkImage image{}; VmaAllocation allocation{}; VkImageView view{}; };
static_assert(sizeof(SceneUniforms) == 144 && sizeof(ModelUniforms) == 160);
static_assert(sizeof(PointGpu) == 32 && sizeof(SpotGpu) == 64 && sizeof(DrawPush) == 16);

struct PointLight {
    glm::vec3 position{1.5f, 1.2f, 2.0f};
    glm::vec3 color{1.0f, 0.45f, 0.2f};
    float intensity = 3.0f;
};
struct SpotLight {
    glm::vec3 position{-1.2f, 1.0f, 1.7f};
    glm::vec3 direction{1.2f, -1.0f, -1.7f};
    glm::vec3 color{0.25f, 0.55f, 1.0f};
    float intensity = 8.0f;
    float inner_angle = 15.0f;
    float outer_angle = 25.0f;
};
struct State {
    VkDescriptorSetLayout scene_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout model_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet scene_set = VK_NULL_HANDLE;
    std::array<VkDescriptorSet,2> material_sets{};
    std::array<Texture,5> textures{};
    VkSampler sampler{};
    size_t model_stride=0, model_capacity=0;
    uint32_t next_model=2;
    std::array<int,2> material_indices{0,1};
    std::array<bool,2> effects{true,false};
    int selected_model=0;
    float time=0;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline lit_pipeline = VK_NULL_HANDLE;
    VkPipeline marker_pipeline = VK_NULL_HANDLE;
    VkPipeline line_pipeline = VK_NULL_HANDLE;
    Buffer scene_ubo, model_ubo, point_ssbo, spot_ssbo;
    size_t point_capacity = 0, spot_capacity = 0;
    Mesh icosahedron, cube, cone;
    glm::vec3 camera_position{-0.6f, 0.0f, 7.5f};
    glm::vec3 camera_rotation{0.0f};
    float field_of_view = 48.0f, near_plane = 0.1f, far_plane = 100.0f;
    glm::vec3 model_position{0.0f}, model_rotation{12.0f, 20.0f, 0.0f}, model_scale{1.0f};
    glm::vec3 albedo{0.65f, 0.70f, 0.78f}, specular{0.4f};
    float shininess = 64.0f;
    glm::vec3 ambient_color{1.0f};
    float ambient_intensity = 0.08f;
    glm::vec3 directional_direction{-0.6f, -1.0f, -0.8f}, directional_color{1.0f};
    float directional_intensity = 0.65f;
    std::vector<PointLight> points{PointLight{}};
    std::vector<SpotLight> spots{SpotLight{}};
    bool show_lights = true;
    std::string error;
} app;

void destroyBuffer(Buffer& buffer) {
    if (buffer.handle) vmaDestroyBuffer(graphics::internal::context.allocator,
                                       buffer.handle, buffer.allocation);
    buffer = {};
}
void destroyResources() {
    auto& c = graphics::internal::context;
    if (!c.device) return;
    vkDeviceWaitIdle(c.device);
    for (VkPipeline p : {app.lit_pipeline, app.marker_pipeline, app.line_pipeline})
        if (p) vkDestroyPipeline(c.device, p, nullptr);
    if (app.pipeline_layout) vkDestroyPipelineLayout(c.device, app.pipeline_layout, nullptr);
    if (app.descriptor_pool) vkDestroyDescriptorPool(c.device, app.descriptor_pool, nullptr);
    if(app.sampler) vkDestroySampler(c.device,app.sampler,nullptr);
    for(auto& t:app.textures) { if(t.view) vkDestroyImageView(c.device,t.view,nullptr); if(t.image) vmaDestroyImage(c.allocator,t.image,t.allocation); }
    if (app.scene_layout) vkDestroyDescriptorSetLayout(c.device, app.scene_layout, nullptr);
    if (app.model_layout) vkDestroyDescriptorSetLayout(c.device, app.model_layout, nullptr);
    for (Buffer* b : {&app.scene_ubo, &app.model_ubo, &app.point_ssbo, &app.spot_ssbo,
                     &app.icosahedron.vertices, &app.icosahedron.indices,
                     &app.cube.vertices, &app.cube.indices, &app.cone.vertices, &app.cone.indices})
        destroyBuffer(*b);
}
bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer) {
    auto& c = graphics::internal::context;
    const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    const VmaAllocationCreateInfo memory{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO, .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT};
    VmaAllocationInfo result{};
    if (vmaCreateBuffer(c.allocator, &info, &memory, &buffer.handle,
                        &buffer.allocation, &result) != VK_SUCCESS) return false;
    buffer.mapped = result.pMappedData;
    buffer.size = size;
    return true;
}
void writeBuffer(Buffer& buffer, const void* data, size_t bytes) {
    if (!bytes) return;
    std::memcpy(buffer.mapped, data, bytes);
    vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, bytes);
}
// Одноразовая загрузка на GPU: команды завершаются до освобождения staging-буфера.
VkCommandBuffer beginUpload(VkCommandPool& pool) {
    auto& c=graphics::internal::context;
    const VkCommandPoolCreateInfo pi{.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=c.graphics_queue_index};
    if(vkCreateCommandPool(c.device,&pi,nullptr,&pool)!=VK_SUCCESS)return VK_NULL_HANDLE;
    const VkCommandBufferAllocateInfo ai{.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    VkCommandBuffer cmd{};vkAllocateCommandBuffers(c.device,&ai,&cmd);
    const VkCommandBufferBeginInfo bi{.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cmd,&bi);return cmd;
}
void endUpload(VkCommandBuffer cmd,VkCommandPool pool) {
    auto& c=graphics::internal::context;vkEndCommandBuffer(cmd);
    const VkSubmitInfo submit{.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};
    vkQueueSubmit(c.graphics_queue,1,&submit,VK_NULL_HANDLE);vkQueueWaitIdle(c.graphics_queue);vkDestroyCommandPool(c.device,pool,nullptr);
}
bool uploadDeviceBuffer(Buffer& output,const void* data,size_t size,VkBufferUsageFlags usage) {
    Buffer staging;if(!createBuffer(size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,staging))return false;
    writeBuffer(staging,data,size);
    auto& c=graphics::internal::context;
    const VkBufferCreateInfo info{.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=size,.usage=usage|VK_BUFFER_USAGE_TRANSFER_DST_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    const VmaAllocationCreateInfo memory{.usage=VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,.requiredFlags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    if(vmaCreateBuffer(c.allocator,&info,&memory,&output.handle,&output.allocation,nullptr)!=VK_SUCCESS){destroyBuffer(staging);return false;}
    output.size=size;
    VkCommandPool pool{};auto cmd=beginUpload(pool);if(!cmd){destroyBuffer(staging);return false;}
    const VkBufferCopy copy{0,0,size};vkCmdCopyBuffer(cmd,staging.handle,output.handle,1,&copy);
    const VkBufferMemoryBarrier barrier{.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask=usage==VK_BUFFER_USAGE_VERTEX_BUFFER_BIT?VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT:VK_ACCESS_INDEX_READ_BIT,
        .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.buffer=output.handle,.size=size};
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,0,0,nullptr,1,&barrier,0,nullptr);
    endUpload(cmd,pool);destroyBuffer(staging);return true;
}
bool loadTexture(Texture& texture,const char* filename,bool srgb) {
    int width,height,channels;
    auto pixels=stbi_load((std::string(APP_TEXTURE_DIR)+"/"+filename).c_str(),&width,&height,&channels,4);
    if(!pixels){std::cerr<<"Texture decode failed: "<<filename<<"\n";return false;}
    Buffer staging;const size_t bytes=static_cast<size_t>(width)*height*4;
    if(!createBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,staging)){stbi_image_free(pixels);return false;}
    writeBuffer(staging,pixels,bytes);stbi_image_free(pixels);
    auto& c=graphics::internal::context;
    const VkFormat format=srgb?VK_FORMAT_R8G8B8A8_SRGB:VK_FORMAT_R8G8B8A8_UNORM;
    const VkImageCreateInfo image{.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_2D,.format=format,
        .extent={static_cast<uint32_t>(width),static_cast<uint32_t>(height),1},.mipLevels=1,.arrayLayers=1,.samples=VK_SAMPLE_COUNT_1_BIT,
        .tiling=VK_IMAGE_TILING_OPTIMAL,.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE,.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED};
    const VmaAllocationCreateInfo memory{.usage=VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,.requiredFlags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    if(vmaCreateImage(c.allocator,&image,&memory,&texture.image,&texture.allocation,nullptr)!=VK_SUCCESS){destroyBuffer(staging);return false;}
    VkCommandPool pool{};auto cmd=beginUpload(pool);if(!cmd){destroyBuffer(staging);return false;}
    VkImageMemoryBarrier barrier{.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
        .image=texture.image,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
    const VkBufferImageCopy region{.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent=image.extent};
    vkCmdCopyBufferToImage(cmd,staging.handle,texture.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&region);
    barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;barrier.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
    endUpload(cmd,pool);destroyBuffer(staging);
    const VkImageViewCreateInfo view{.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,.image=texture.image,.viewType=VK_IMAGE_VIEW_TYPE_2D,.format=format,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
    return vkCreateImageView(c.device,&view,nullptr,&texture.view)==VK_SUCCESS;
}
bool createTextures() {
    const VkSamplerCreateInfo sampler{.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,.magFilter=VK_FILTER_LINEAR,.minFilter=VK_FILTER_LINEAR,
        .mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST,.addressModeU=VK_SAMPLER_ADDRESS_MODE_REPEAT,.addressModeV=VK_SAMPLER_ADDRESS_MODE_REPEAT,.addressModeW=VK_SAMPLER_ADDRESS_MODE_REPEAT};
    if(vkCreateSampler(graphics::internal::context.device,&sampler,nullptr,&app.sampler)!=VK_SUCCESS)return false;
    const char* files[]{"tiles.png","bricks.png","specular.png","emissive.png","black.png"};
    for(int i=0;i<5;++i)if(!loadTexture(app.textures[i],files[i],i<2||i==3))return false;
    return true;
}
bool createMesh(Mesh& mesh,const std::vector<Vertex>& vertices,const std::vector<uint32_t>& indices) {
    if(!uploadDeviceBuffer(mesh.vertices,vertices.data(),vertices.size()*sizeof(Vertex),VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)||
       !uploadDeviceBuffer(mesh.indices,indices.data(),indices.size()*sizeof(uint32_t),VK_BUFFER_USAGE_INDEX_BUFFER_BIT))return false;
    mesh.index_count=static_cast<uint32_t>(indices.size());return true;
}
bool createMeshes() {
    const float phi = (1.0f + std::sqrt(5.0f)) / 2.0f;
    std::array<glm::vec3, 12> positions{{
        {-1,phi,0},{1,phi,0},{-1,-phi,0},{1,-phi,0},
        {0,-1,phi},{0,1,phi},{0,-1,-phi},{0,1,-phi},
        {phi,0,-1},{phi,0,1},{-phi,0,-1},{-phi,0,1}}};
    for (auto& p : positions) p = glm::normalize(p);
    const std::array<uint32_t, 60> faces{{
        0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
        1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
        3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
        4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1}};
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    for (size_t i = 0; i < faces.size(); i += 3) {
        glm::vec3 a = positions[faces[i]], b = positions[faces[i+1]], c = positions[faces[i+2]];
        if (glm::dot(glm::cross(b-a,c-a), a+b+c) < 0) std::swap(b,c);
        // Одна нормаль на грань сохраняет плоские грани правильного икосаэдра.
        const glm::vec3 normal = glm::normalize(glm::cross(b-a,c-a));
        for (auto p : {a,b,c}) {
            indices.push_back(static_cast<uint32_t>(vertices.size()));
            const glm::vec2 uv=vertices.size()%3==0?glm::vec2(0,0):(vertices.size()%3==1?glm::vec2(1,0):glm::vec2(0.5f,1));
            vertices.push_back({p, normal, uv});
        }
    }
    if (!createMesh(app.icosahedron, vertices, indices)) return false;
    vertices.clear(); indices.clear();
    for (glm::vec3 p : std::array<glm::vec3,8>{{
        {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
        {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}}}) vertices.push_back({p,{0,0,0}});
    indices = {0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,
               3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5};
    if (!createMesh(app.cube, vertices, indices)) return false;
    // Единичный проволочный конус вдоль +Z: окружность и четыре образующие.
    vertices = {{{0,0,0},{0,0,0}}}; indices.clear();
    constexpr uint32_t segments = 32;
    for (uint32_t i=0; i<segments; ++i) {
        const float angle = 2.0f * std::numbers::pi_v<float> * i / segments;
        vertices.push_back({{std::cos(angle),std::sin(angle),1.0f},{0,0,0}});
        indices.push_back(i+1); indices.push_back((i+1)%segments+1);
        if (i%8==0) { indices.push_back(0); indices.push_back(i+1); }
    }
    return createMesh(app.cone, vertices, indices);
}

void bindStorage(uint32_t binding, const Buffer& buffer) {
    const VkDescriptorBufferInfo info{buffer.handle,0,buffer.size};
    const VkWriteDescriptorSet write{.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet=app.scene_set,.dstBinding=binding,.descriptorCount=1,
        .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&info};
    vkUpdateDescriptorSets(graphics::internal::context.device,1,&write,0,nullptr);
}
// Размер SSBO растёт по необходимости. В GLSL используются массивы без заданной длины.
bool ensureStorage(Buffer& buffer, size_t& capacity, size_t count, size_t element_size, uint32_t binding) {
    if (buffer.handle && count<=capacity) return true;
    size_t next = std::max<size_t>(1,capacity);
    while (next<count) next*=2;
    Buffer replacement;
    if (!createBuffer(next*element_size,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,replacement)) return false;
    destroyBuffer(buffer);
    buffer=replacement; capacity=next;
    bindStorage(binding,buffer);
    return true;
}
void bindModelBuffers() {
    const VkDescriptorBufferInfo info{app.model_ubo.handle,0,sizeof(ModelUniforms)};
    for(auto set:app.material_sets) {
        const VkWriteDescriptorSet write{.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,
            .dstBinding=0,.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,.pBufferInfo=&info};
        vkUpdateDescriptorSets(graphics::internal::context.device,1,&write,0,nullptr);
    }
}
bool ensureModels(size_t count) {
    if(count<=app.model_capacity) return true;
    size_t capacity=std::max<size_t>(4,app.model_capacity*2);
    while(capacity<count) capacity*=2;
    Buffer replacement;
    if(!createBuffer(capacity*app.model_stride,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,replacement)) return false;
    destroyBuffer(app.model_ubo);app.model_ubo=replacement;app.model_capacity=capacity;bindModelBuffers();return true;
}
bool createDescriptors() {
    auto device=graphics::internal::context.device;
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device,&properties);
    size_t alignment=properties.limits.minUniformBufferOffsetAlignment;
    app.model_stride=(sizeof(ModelUniforms)+alignment-1)/alignment*alignment;
    const VkDescriptorSetLayoutBinding scene_bindings[]{
        {0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
        {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
        {2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}};
    const VkDescriptorSetLayoutCreateInfo scene_info{.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=3,.pBindings=scene_bindings};
    if(vkCreateDescriptorSetLayout(device,&scene_info,nullptr,&app.scene_layout)!=VK_SUCCESS) return false;
    const VkDescriptorSetLayoutBinding bindings[]{
        {0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,1,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
        {1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
        {2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr},
        {3,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr}};
    const VkDescriptorSetLayoutCreateInfo model_info{.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=4,.pBindings=bindings};
    if(vkCreateDescriptorSetLayout(device,&model_info,nullptr,&app.model_layout)!=VK_SUCCESS) return false;
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,2},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,6}};
    const VkDescriptorPoolCreateInfo pool{.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,.maxSets=3,.poolSizeCount=4,.pPoolSizes=sizes};
    if(vkCreateDescriptorPool(device,&pool,nullptr,&app.descriptor_pool)!=VK_SUCCESS) return false;
    const VkDescriptorSetLayout layouts[]{app.scene_layout,app.model_layout,app.model_layout};
    const VkDescriptorSetAllocateInfo allocate{.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,.descriptorPool=app.descriptor_pool,.descriptorSetCount=3,.pSetLayouts=layouts};
    VkDescriptorSet sets[3]{};
    if(vkAllocateDescriptorSets(device,&allocate,sets)!=VK_SUCCESS) return false;
    app.scene_set=sets[0];app.material_sets={sets[1],sets[2]};
    if(!createBuffer(sizeof(SceneUniforms),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,app.scene_ubo)||!ensureModels(4))return false;
    const VkDescriptorBufferInfo scene_buffer{app.scene_ubo.handle,0,sizeof(SceneUniforms)};
    const VkWriteDescriptorSet write{.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=app.scene_set,.dstBinding=0,
        .descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,.pBufferInfo=&scene_buffer};
    vkUpdateDescriptorSets(device,1,&write,0,nullptr);
    // Пункт 3: у первого материала карты блика/свечения, у второго — чёрная заглушка.
    const int maps[2][3]={{0,2,3},{1,4,4}};
    for(int m=0;m<2;++m)for(int j=0;j<3;++j){
        const VkDescriptorImageInfo image{app.sampler,app.textures[maps[m][j]].view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        const VkWriteDescriptorSet image_write{.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=app.material_sets[m],
            .dstBinding=static_cast<uint32_t>(j+1),.descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.pImageInfo=&image};
        vkUpdateDescriptorSets(device,1,&image_write,0,nullptr);
    }
    return ensureStorage(app.point_ssbo,app.point_capacity,app.points.size(),sizeof(PointGpu),1)&&
        ensureStorage(app.spot_ssbo,app.spot_capacity,app.spots.size(),sizeof(SpotGpu),2);
}
VkShaderModule loadShader(const char* name) {
    const std::string path=std::string(APP_SHADER_DIR)+"/"+name+".spv";
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if (!file) return VK_NULL_HANDLE;
    const auto size=file.tellg();
    if (size<=0 || size%4!=0) return VK_NULL_HANDLE;
    std::vector<uint32_t> code(static_cast<size_t>(size)/4);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(code.data()),size)) return VK_NULL_HANDLE;
    const VkShaderModuleCreateInfo info{.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize=static_cast<size_t>(size),.pCode=code.data()};
    VkShaderModule module=VK_NULL_HANDLE;
    vkCreateShaderModule(graphics::internal::context.device,&info,nullptr,&module);
    return module;
}
bool createPipeline(const char* vertex_name, const char* fragment_name,
                    VkPrimitiveTopology topology, bool lit, VkPipeline& output) {
    auto& c=graphics::internal::context;
    VkShaderModule vertex=loadShader(vertex_name), fragment=loadShader(fragment_name);
    if (!vertex || !fragment) {
        if(vertex) vkDestroyShaderModule(c.device,vertex,nullptr);
        if(fragment) vkDestroyShaderModule(c.device,fragment,nullptr);
        return false;
    }
    const VkPipelineShaderStageCreateInfo stages[]{
        {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_VERTEX_BIT,.module=vertex,.pName="main"},
        {.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_FRAGMENT_BIT,.module=fragment,.pName="main"}};
    const VkVertexInputBindingDescription binding{0,sizeof(Vertex),VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription attributes[]{
        {0,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,position)},
        {1,0,VK_FORMAT_R32G32B32_SFLOAT,offsetof(Vertex,normal)},
        {2,0,VK_FORMAT_R32G32_SFLOAT,offsetof(Vertex,uv)}};
    const VkPipelineVertexInputStateCreateInfo input{.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount=1,.pVertexBindingDescriptions=&binding,
        .vertexAttributeDescriptionCount=lit?3u:1u,.pVertexAttributeDescriptions=attributes};
    const VkPipelineInputAssemblyStateCreateInfo assembly{.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,.topology=topology};
    const VkPipelineViewportStateCreateInfo viewport{.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,.viewportCount=1,.scissorCount=1};
    const VkPipelineRasterizationStateCreateInfo raster{.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode=VK_POLYGON_MODE_FILL,.cullMode=lit?VK_CULL_MODE_BACK_BIT:VK_CULL_MODE_NONE,
        .frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,.lineWidth=1.0f};
    const VkPipelineMultisampleStateCreateInfo samples{.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
    const VkPipelineDepthStencilStateCreateInfo depth{.sType=VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable=VK_TRUE,.depthWriteEnable=VK_TRUE,.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL};
    const VkPipelineColorBlendAttachmentState attachment{.colorWriteMask=VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT};
    const VkPipelineColorBlendStateCreateInfo blend{.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,.attachmentCount=1,.pAttachments=&attachment};
    const VkDynamicState states[]{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic{.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,.dynamicStateCount=2,.pDynamicStates=states};
    const VkGraphicsPipelineCreateInfo info{.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount=2,.pStages=stages,.pVertexInputState=&input,.pInputAssemblyState=&assembly,
        .pViewportState=&viewport,.pRasterizationState=&raster,.pMultisampleState=&samples,
        .pDepthStencilState=&depth,.pColorBlendState=&blend,.pDynamicState=&dynamic,
        .layout=app.pipeline_layout,.renderPass=c.render_pass,.subpass=0};
    const auto result=vkCreateGraphicsPipelines(c.device,VK_NULL_HANDLE,1,&info,nullptr,&output);
    vkDestroyShaderModule(c.device,vertex,nullptr); vkDestroyShaderModule(c.device,fragment,nullptr);
    return result==VK_SUCCESS;
}
bool createPipelines() {
    const VkDescriptorSetLayout layouts[]{app.scene_layout,app.model_layout};
    const VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(DrawPush)};
    const VkPipelineLayoutCreateInfo info{.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=2,.pSetLayouts=layouts,.pushConstantRangeCount=1,.pPushConstantRanges=&push};
    if (vkCreatePipelineLayout(graphics::internal::context.device,&info,nullptr,&app.pipeline_layout)!=VK_SUCCESS) return false;
    return createPipeline("lighting.vert","lighting.frag",VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,true,app.lit_pipeline) &&
           createPipeline("unlit.vert","unlit.frag",VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,false,app.marker_pipeline) &&
           createPipeline("unlit.vert","unlit.frag",VK_PRIMITIVE_TOPOLOGY_LINE_LIST,false,app.line_pipeline);
}

glm::vec3 safeDirection(glm::vec3 v) {
    return glm::dot(v,v)>1e-8f?glm::normalize(v):glm::vec3(0,0,-1);
}
glm::mat4 transform(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale) {
    glm::mat4 m=glm::translate(glm::mat4(1),position);
    m=glm::rotate(m,glm::radians(rotation.x),glm::vec3(1,0,0));
    m=glm::rotate(m,glm::radians(rotation.y),glm::vec3(0,1,0));
    m=glm::rotate(m,glm::radians(rotation.z),glm::vec3(0,0,1));
    return glm::scale(m,scale);
}
bool updateGpu() {
    // render() вызывается после ожидания fence: SSBO можно безопасно увеличить и заполнить.
    if (!ensureStorage(app.point_ssbo,app.point_capacity,app.points.size(),sizeof(PointGpu),1) ||
        !ensureStorage(app.spot_ssbo,app.spot_capacity,app.spots.size(),sizeof(SpotGpu),2)) {
        app.error="Failed to allocate light storage buffer";
        return false;
    }
    std::vector<PointGpu> points;
    for (const auto& p:app.points) points.push_back({glm::vec4(p.position,1),glm::vec4(p.color,p.intensity)});
    std::vector<SpotGpu> spots;
    for (const auto& s:app.spots) spots.push_back({glm::vec4(s.position,1),glm::vec4(safeDirection(s.direction),0),
        glm::vec4(s.color,s.intensity),{std::cos(glm::radians(s.inner_angle)),std::cos(glm::radians(s.outer_angle)),0,0}});
    writeBuffer(app.point_ssbo,points.data(),points.size()*sizeof(PointGpu));
    writeBuffer(app.spot_ssbo,spots.data(),spots.size()*sizeof(SpotGpu));
    const auto& extent=graphics::internal::context.swapchain_extent;
    const float aspect=static_cast<float>(extent.width)/std::max(1u,extent.height);
    glm::mat4 projection=glm::perspective(glm::radians(app.field_of_view),aspect,app.near_plane,app.far_plane);
    projection[1][1]*=-1;
    const glm::mat4 view=glm::inverse(transform(app.camera_position,app.camera_rotation,glm::vec3(1)));
    const SceneUniforms scene{projection*view,glm::vec4(app.camera_position,1),
        glm::vec4(app.ambient_color,app.ambient_intensity),glm::vec4(safeDirection(app.directional_direction),0),
        glm::vec4(app.directional_color,app.directional_intensity),
        {static_cast<uint32_t>(points.size()),static_cast<uint32_t>(spots.size()),0,0}};
    if(!ensureModels(2+app.points.size()+2*app.spots.size()))return false;
    app.next_model=2;
    for(int i=0;i<2;++i){
    const glm::mat4 model=transform(app.model_position+glm::vec3(i?1.2f:-1.2f,0,0),app.model_rotation,app.model_scale);
    // Обратная транспонированная матрица сохраняет нормали при неравномерном масштабе.
    const ModelUniforms material{model,glm::transpose(glm::inverse(model)),glm::vec4(app.albedo,1),glm::vec4(app.specular,app.shininess)};
    std::memcpy(static_cast<char*>(app.model_ubo.mapped)+i*app.model_stride,&material,sizeof(material));
    }
    vmaFlushAllocation(graphics::internal::context.allocator,app.model_ubo.allocation,0,VK_WHOLE_SIZE);
    writeBuffer(app.scene_ubo,&scene,sizeof(scene));
    app.error.clear();
    return true;
}
void drawMesh(VkCommandBuffer cmd, const Mesh& mesh) {
    const VkDeviceSize offset=0;
    vkCmdBindVertexBuffers(cmd,0,1,&mesh.vertices.handle,&offset);
    vkCmdBindIndexBuffer(cmd,mesh.indices.handle,0,VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd,mesh.index_count,1,0,0,0);
}
void drawUnlit(VkCommandBuffer cmd, const Mesh& mesh, glm::mat4 model, glm::vec3 color) {
    const uint32_t row=app.next_model++;
    const ModelUniforms data{model,glm::mat4(1),glm::vec4(color,1),glm::vec4(0)};
    std::memcpy(static_cast<char*>(app.model_ubo.mapped)+row*app.model_stride,&data,sizeof(data));
    vmaFlushAllocation(graphics::internal::context.allocator,app.model_ubo.allocation,row*app.model_stride,sizeof(data));
    const uint32_t offset=static_cast<uint32_t>(row*app.model_stride);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,app.pipeline_layout,1,1,&app.material_sets[0],1,&offset);
    drawMesh(cmd,mesh);
}
void drawLightMarkers(VkCommandBuffer cmd) {
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,app.marker_pipeline);
    for (const auto& p:app.points)
        drawUnlit(cmd,app.cube,transform(p.position,{0,0,0},glm::vec3(0.07f)),p.color);
    for (const auto& s:app.spots)
        drawUnlit(cmd,app.cube,transform(s.position,{0,0,0},glm::vec3(0.07f)),s.color);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,app.line_pipeline);
    for (const auto& s:app.spots) {
        const glm::vec3 forward=safeDirection(s.direction);
        const glm::vec3 up=std::abs(forward.y)>0.99f?glm::vec3(1,0,0):glm::vec3(0,1,0);
        const glm::vec3 right=glm::normalize(glm::cross(up,forward));
        glm::mat4 basis(1);
        basis[0]=glm::vec4(right,0); basis[1]=glm::vec4(glm::cross(forward,right),0); basis[2]=glm::vec4(forward,0);
        constexpr float length=2.0f;
        const float radius=length*std::tan(glm::radians(s.outer_angle));
        const auto model=glm::translate(glm::mat4(1),s.position)*basis*glm::scale(glm::mat4(1),glm::vec3(radius,radius,length));
        drawUnlit(cmd,app.cone,model,glm::vec3(1));
    }
}
void drawControls() {
    ImGui::SetNextWindowPos({12,12},ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({365,680},ImGuiCond_FirstUseEver);
    ImGui::Begin("Lab 3 - Variant 11");
    ImGui::PushItemWidth(155.0f);
    if (!app.error.empty()) ImGui::TextWrapped("%s",app.error.c_str());
    if (ImGui::CollapsingHeader("Camera",ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Position##camera",&app.camera_position.x,0.03f);
        ImGui::DragFloat3("Rotation XYZ (deg)##camera",&app.camera_rotation.x,0.5f,-180,180,"%.1f");
        ImGui::SliderFloat("Field of view",&app.field_of_view,15,100,"%.1f deg",ImGuiSliderFlags_AlwaysClamp);
        ImGui::DragFloat("Near plane",&app.near_plane,0.01f,0.01f,app.far_plane-0.01f);
        ImGui::DragFloat("Far plane",&app.far_plane,0.5f,app.near_plane+0.01f,500);
        app.near_plane=std::clamp(app.near_plane,0.01f,499.0f);
        app.far_plane=std::clamp(app.far_plane,app.near_plane+0.01f,500.0f);
    }
    const char* objects[]{"Icosahedron 1","Icosahedron 2"};
    const char* textures[]{"Tiles + maps","Bricks"};
    ImGui::Combo("Object",&app.selected_model,objects,2);
    ImGui::Combo("Texture",&app.material_indices[app.selected_model],textures,2);
    ImGui::Checkbox("Animated sampling",&app.effects[app.selected_model]);
    if (ImGui::CollapsingHeader("Model and material")) {
        ImGui::DragFloat3("Position##model",&app.model_position.x,0.02f);
        ImGui::DragFloat3("Rotation XYZ##model",&app.model_rotation.x,1,-180,180);
        ImGui::DragFloat3("Scale",&app.model_scale.x,0.01f,0.1f,3,"%.2f",ImGuiSliderFlags_AlwaysClamp);
        ImGui::ColorEdit3("Albedo",&app.albedo.x);
        ImGui::ColorEdit3("Specular color",&app.specular.x);
        ImGui::SliderFloat("Shininess",&app.shininess,1,256,"%.0f",ImGuiSliderFlags_AlwaysClamp);
    }
    if (ImGui::CollapsingHeader("Ambient and directional",ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient color",&app.ambient_color.x);
        ImGui::SliderFloat("Ambient intensity",&app.ambient_intensity,0,1,"%.3f",ImGuiSliderFlags_AlwaysClamp);
        ImGui::DragFloat3("Direction",&app.directional_direction.x,0.02f);
        ImGui::ColorEdit3("Directional color",&app.directional_color.x);
        ImGui::SliderFloat("Directional intensity",&app.directional_intensity,0,5,"%.3f",ImGuiSliderFlags_AlwaysClamp);
    }
    ImGui::Checkbox("Show light cubes and spot cones",&app.show_lights);
    if (ImGui::CollapsingHeader("Point lights",ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Add point light")) app.points.emplace_back();
        for (size_t i=0;i<app.points.size();) {
            ImGui::PushID(static_cast<int>(i));
            const bool open=ImGui::TreeNode("point","Point %zu",i+1);
            bool remove=false;
            if (open) {
                auto& p=app.points[i];
                ImGui::DragFloat3("Position",&p.position.x,0.03f);
                ImGui::ColorEdit3("Color",&p.color.x);
                ImGui::SliderFloat("Intensity",&p.intensity,0,40,"%.3f",ImGuiSliderFlags_AlwaysClamp);
                remove=ImGui::Button("Remove point");
                ImGui::TreePop();
            }
            ImGui::PopID();
            if (remove) app.points.erase(app.points.begin()+static_cast<std::ptrdiff_t>(i)); else ++i;
        }
    }
    if (ImGui::CollapsingHeader("Spot lights",ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Add spot light")) app.spots.emplace_back();
        for (size_t i=0;i<app.spots.size();) {
            ImGui::PushID(static_cast<int>(i));
            const bool open=ImGui::TreeNode("spot","Spot %zu",i+1);
            bool remove=false;
            if (open) {
                auto& s=app.spots[i];
                ImGui::DragFloat3("Position",&s.position.x,0.03f);
                ImGui::DragFloat3("Direction",&s.direction.x,0.02f);
                ImGui::ColorEdit3("Color",&s.color.x);
                ImGui::SliderFloat("Intensity",&s.intensity,0,60,"%.3f",ImGuiSliderFlags_AlwaysClamp);
                ImGui::SliderFloat("Inner half-angle",&s.inner_angle,0,80,"%.1f deg");
                ImGui::SliderFloat("Outer half-angle",&s.outer_angle,s.inner_angle,80,"%.1f deg");
                s.inner_angle=std::clamp(s.inner_angle,0.0f,80.0f);
                s.outer_angle=std::clamp(s.outer_angle,s.inner_angle,80.0f);
                remove=ImGui::Button("Remove spot");
                ImGui::TreePop();
            }
            ImGui::PopID();
            if (remove) app.spots.erase(app.spots.begin()+static_cast<std::ptrdiff_t>(i)); else ++i;
        }
    }
    ImGui::PopItemWidth();
    ImGui::End();
}
} // namespace

bool initialize() {
    if (!createTextures() || !createMeshes() || !createDescriptors() || !createPipelines()) {
        std::cerr<<"Lab 3 initialization failed\n";
        destroyResources(); return false;
    }
    return true;
}
void shutdown() { destroyResources(); }
void update(double time) { app.time=static_cast<float>(time);drawControls(); }
void render(const graphics::internal::FrameData& fd) {
    if (!fd.command_buffer || !fd.framebuffer) return;
    const bool ready=updateGpu();
    const auto cmd=fd.command_buffer;
    vkResetCommandBuffer(cmd,0);
    const VkCommandBufferBeginInfo begin{.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cmd,&begin);
    const VkClearValue clear[]{{.color={.float32={0.025f,0.035f,0.055f,1}}},{.depthStencil={1,0}}};
    const auto& c=graphics::internal::context;
    const VkRenderPassBeginInfo pass{.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass=c.render_pass,.framebuffer=fd.framebuffer,.renderArea={.extent=c.swapchain_extent},
        .clearValueCount=2,.pClearValues=clear};
    vkCmdBeginRenderPass(cmd,&pass,VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0,0,static_cast<float>(c.swapchain_extent.width),static_cast<float>(c.swapchain_extent.height),0,1};
    const VkRect2D scissor{{0,0},c.swapchain_extent};
    vkCmdSetViewport(cmd,0,1,&viewport); vkCmdSetScissor(cmd,0,1,&scissor);
    if (ready) {
        const VkDescriptorSet sets[]{app.scene_set,app.material_sets[0]};
        const uint32_t zero=0;
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,app.pipeline_layout,0,2,sets,1,&zero);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,app.lit_pipeline);
        for(uint32_t i=0;i<2;++i){
            const uint32_t offset=static_cast<uint32_t>(i*app.model_stride);
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,app.pipeline_layout,1,1,&app.material_sets[app.material_indices[i]],1,&offset);
            const DrawPush push{{app.effects[i]?1.0f:0.0f,app.time,0,0}};
            vkCmdPushConstants(cmd,app.pipeline_layout,VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(push),&push);
            drawMesh(cmd,app.icosahedron);
        }
        if (app.show_lights) drawLightMarkers(cmd);
    }
    vkCmdEndRenderPass(cmd); vkEndCommandBuffer(cmd);
}
} // namespace application
