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

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#ifndef APP_SHADER_DIR
#define APP_SHADER_DIR "shaders"
#endif

namespace application {

namespace {

constexpr uint32_t object_count = 2;
constexpr float pi = std::numbers::pi_v<float>;

struct Vertex {
	glm::vec3 position;
	glm::vec3 color;
};

struct GlobalUniforms {
	glm::mat4 view;
	glm::mat4 projection;
};

struct ObjectUniforms {
	glm::mat4 model;
	glm::vec4 tint;
};

struct Buffer {
	VkBuffer handle = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
};

struct Object {
	glm::vec3 position{};
	glm::vec3 rotation_degrees{};
	glm::vec3 scale{0.8f};
	glm::vec3 tint{1.0f};
	float phase = 0.0f;
	Buffer uniforms;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

struct AppState {
	VkDescriptorSetLayout global_set_layout = VK_NULL_HANDLE;
	VkDescriptorSetLayout object_set_layout = VK_NULL_HANDLE;
	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
	VkDescriptorSet global_set = VK_NULL_HANDLE;
	Buffer global_uniforms;
	Buffer vertex_buffer;
	Buffer index_buffer;
	uint32_t index_count = 0;
	std::array<Object, object_count> objects{};
	uint32_t selected_object = 0;
	bool orthographic = false;
	bool animation_playing = true;
	float animation_speed = 0.75f;
	float orbit_radius = 0.28f;
	float vertical_amplitude = 0.14f;
	float animation_time = 0.0f;
	double previous_update_time = 0.0;
	bool has_previous_update_time = false;
} app;

void destroyBuffer(Buffer& buffer) {
	if (buffer.handle != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, buffer.handle,
		                 buffer.allocation);
	}
	buffer = {};
}

void destroyResources() {
	auto& context = graphics::internal::context;
	if (context.device == VK_NULL_HANDLE) {
		return;
	}
	if (context.graphics_queue != VK_NULL_HANDLE) {
		vkQueueWaitIdle(context.graphics_queue);
	}

	if (app.pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, app.pipeline, nullptr);
		app.pipeline = VK_NULL_HANDLE;
	}
	if (app.pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, app.pipeline_layout, nullptr);
		app.pipeline_layout = VK_NULL_HANDLE;
	}
	if (app.descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, app.descriptor_pool, nullptr);
		app.descriptor_pool = VK_NULL_HANDLE;
	}
	if (app.global_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, app.global_set_layout, nullptr);
		app.global_set_layout = VK_NULL_HANDLE;
	}
	if (app.object_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, app.object_set_layout, nullptr);
		app.object_set_layout = VK_NULL_HANDLE;
	}

	destroyBuffer(app.global_uniforms);
	for (Object& object : app.objects) {
		destroyBuffer(object.uniforms);
		object.descriptor_set = VK_NULL_HANDLE;
	}
	destroyBuffer(app.vertex_buffer);
	destroyBuffer(app.index_buffer);
	app.global_set = VK_NULL_HANDLE;
	app.index_count = 0;
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& result) {
	auto& context = graphics::internal::context;
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
		.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
	};
	VmaAllocationInfo allocation_result{};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &result.handle, &result.allocation, &allocation_result) != VK_SUCCESS) {
		std::cerr << "Не удалось создать Vulkan-буфер размером " << size << " байт\n";
		return false;
	}
	result.mapped = allocation_result.pMappedData;
	return true;
}

bool uploadBuffer(Buffer& buffer, const void* data, VkDeviceSize size) {
	if (!createBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                 buffer)) {
		return false;
	}
	std::memcpy(buffer.mapped, data, static_cast<size_t>(size));
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size);
	return true;
}

bool uploadIndexBuffer(const std::vector<uint32_t>& indices) {
	if (!createBuffer(indices.size() * sizeof(uint32_t),
	                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
	                  app.index_buffer)) {
		return false;
	}
	const VkDeviceSize bytes = indices.size() * sizeof(uint32_t);
	std::memcpy(app.index_buffer.mapped, indices.data(), static_cast<size_t>(bytes));
	vmaFlushAllocation(graphics::internal::context.allocator, app.index_buffer.allocation,
	                   0, bytes);
	app.index_count = static_cast<uint32_t>(indices.size());
	return true;
}

bool createUniformBuffer(VkDeviceSize size, Buffer& buffer) {
	if (!createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, buffer)) {
		return false;
	}
	std::memset(buffer.mapped, 0, static_cast<size_t>(size));
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size);
	return true;
}

void writeUniform(Buffer& buffer, const void* data, VkDeviceSize size) {
	std::memcpy(buffer.mapped, data, static_cast<size_t>(size));
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size);
}

bool createIcosahedronMesh() {
	const float golden_ratio = (1.0f + std::sqrt(5.0f)) * 0.5f;
	std::array<glm::vec3, 12> positions = {{
		{-1.0f, golden_ratio, 0.0f}, {1.0f, golden_ratio, 0.0f},
		{-1.0f, -golden_ratio, 0.0f}, {1.0f, -golden_ratio, 0.0f},
		{0.0f, -1.0f, golden_ratio}, {0.0f, 1.0f, golden_ratio},
		{0.0f, -1.0f, -golden_ratio}, {0.0f, 1.0f, -golden_ratio},
		{golden_ratio, 0.0f, -1.0f}, {golden_ratio, 0.0f, 1.0f},
		{-golden_ratio, 0.0f, -1.0f}, {-golden_ratio, 0.0f, 1.0f},
	}};
	for (glm::vec3& position : positions) {
		position = glm::normalize(position);
	}

	std::vector<uint32_t> indices = {
		0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
		1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
		3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
		4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1,
	};
	for (size_t i = 0; i < indices.size(); i += 3) {
		const glm::vec3 a = positions[indices[i + 0]];
		const glm::vec3 b = positions[indices[i + 1]];
		const glm::vec3 c = positions[indices[i + 2]];
		if (glm::dot(glm::cross(b - a, c - a), a + b + c) < 0.0f) {
			std::swap(indices[i + 1], indices[i + 2]);
		}
	}

	std::array<Vertex, 12> vertices{};
	for (size_t i = 0; i < positions.size(); ++i) {
		// Generate a repeating RGB palette procedurally from the vertex number.
		const glm::vec3 color = {
			i % 3 == 0 ? 1.0f : 0.1f,
			i % 3 == 1 ? 1.0f : 0.1f,
			i % 3 == 2 ? 1.0f : 0.1f,
		};
		vertices[i] = { positions[i], color };
	}
	if (!uploadBuffer(app.vertex_buffer, vertices.data(), sizeof(vertices)) ||
	    !uploadIndexBuffer(indices)) {
		return false;
	}
	return true;
}

bool createDescriptorLayouts() {
	auto& context = graphics::internal::context;
	const VkDescriptorSetLayoutBinding global_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo global_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &global_binding,
	};
	if (vkCreateDescriptorSetLayout(context.device, &global_layout, nullptr,
	                                &app.global_set_layout) != VK_SUCCESS) {
		std::cerr << "Не удалось создать layout глобального descriptor set\n";
		return false;
	}

	const VkDescriptorSetLayoutBinding object_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo object_layout = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &object_binding,
	};
	if (vkCreateDescriptorSetLayout(context.device, &object_layout, nullptr,
	                                &app.object_set_layout) != VK_SUCCESS) {
		std::cerr << "Не удалось создать layout descriptor set объекта\n";
		return false;
	}
	return true;
}

bool createDescriptors() {
	auto& context = graphics::internal::context;
	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = object_count + 1,
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = object_count + 1,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
	                           &app.descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Не удалось создать descriptor pool приложения\n";
		return false;
	}

	if (!createUniformBuffer(sizeof(GlobalUniforms), app.global_uniforms)) {
		return false;
	}
	for (Object& object : app.objects) {
		if (!createUniformBuffer(sizeof(ObjectUniforms), object.uniforms)) {
			return false;
		}
	}

	const VkDescriptorSetLayout global_layouts[] = { app.global_set_layout };
	const VkDescriptorSetAllocateInfo global_allocate = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = app.descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = global_layouts,
	};
	if (vkAllocateDescriptorSets(context.device, &global_allocate,
	                              &app.global_set) != VK_SUCCESS) {
		std::cerr << "Не удалось выделить глобальный descriptor set\n";
		return false;
	}

	std::array<VkDescriptorSetLayout, object_count> object_layouts{};
	std::array<VkDescriptorSet, object_count> object_sets{};
	object_layouts.fill(app.object_set_layout);
	const VkDescriptorSetAllocateInfo object_allocate = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = app.descriptor_pool,
		.descriptorSetCount = object_count,
		.pSetLayouts = object_layouts.data(),
	};
	if (vkAllocateDescriptorSets(context.device, &object_allocate,
	                              object_sets.data()) != VK_SUCCESS) {
		std::cerr << "Не удалось выделить descriptor sets объектов\n";
		return false;
	}
	for (uint32_t i = 0; i < object_count; ++i) {
		app.objects[i].descriptor_set = object_sets[i];
	}

	const VkDescriptorBufferInfo global_buffer_info = {
		.buffer = app.global_uniforms.handle,
		.offset = 0,
		.range = sizeof(GlobalUniforms),
	};
	const VkWriteDescriptorSet global_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = app.global_set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &global_buffer_info,
	};
	vkUpdateDescriptorSets(context.device, 1, &global_write, 0, nullptr);

	for (uint32_t i = 0; i < object_count; ++i) {
		const VkDescriptorBufferInfo object_buffer_info = {
			.buffer = app.objects[i].uniforms.handle,
			.offset = 0,
			.range = sizeof(ObjectUniforms),
		};
		const VkWriteDescriptorSet object_write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = app.objects[i].descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &object_buffer_info,
		};
		vkUpdateDescriptorSets(context.device, 1, &object_write, 0, nullptr);
	}
	return true;
}

bool readShader(const std::string& filename, std::vector<char>& data) {
	std::ifstream file(filename, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Не удалось открыть SPIR-V шейдер: " << filename << '\n';
		return false;
	}
	const std::streamsize size = file.tellg();
	if (size <= 0 || size % 4 != 0) {
		std::cerr << "Некорректный размер SPIR-V файла: " << filename << '\n';
		return false;
	}
	data.resize(static_cast<size_t>(size));
	file.seekg(0);
	if (!file.read(data.data(), size)) {
		std::cerr << "Не удалось прочитать SPIR-V шейдер: " << filename << '\n';
		return false;
	}
	return true;
}

VkShaderModule createShaderModule(const std::vector<char>& data) {
	const VkShaderModuleCreateInfo create_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = data.size(),
		.pCode = reinterpret_cast<const uint32_t*>(data.data()),
	};
	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &create_info, nullptr,
	                         &module) != VK_SUCCESS) {
		std::cerr << "Не удалось создать Vulkan shader module\n";
	}
	return module;
}

bool createPipeline() {
	auto& context = graphics::internal::context;
	std::vector<char> vertex_code;
	std::vector<char> fragment_code;
	const std::string shader_directory = APP_SHADER_DIR;
	if (!readShader(shader_directory + "/icosahedron.vert.spv", vertex_code) ||
	    !readShader(shader_directory + "/icosahedron.frag.spv", fragment_code)) {
		return false;
	}
	VkShaderModule vertex_module = createShaderModule(vertex_code);
	VkShaderModule fragment_module = createShaderModule(fragment_code);
	if (vertex_module == VK_NULL_HANDLE || fragment_module == VK_NULL_HANDLE) {
		if (vertex_module != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, vertex_module, nullptr);
		if (fragment_module != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, fragment_module, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_module,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_module,
			.pName = "main",
		},
	};
	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = offsetof(Vertex, position) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = offsetof(Vertex, color) },
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = vertex_attributes,
	};
	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};
	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};
	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};
	const VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};
	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};
	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};
	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};
	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	const VkPipelineDynamicStateCreateInfo dynamic = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};
	const VkDescriptorSetLayout set_layouts[] = {
		app.global_set_layout,
		app.object_set_layout,
	};
	const VkPipelineLayoutCreateInfo pipeline_layout = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = set_layouts,
	};
	if (vkCreatePipelineLayout(context.device, &pipeline_layout, nullptr,
	                           &app.pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Не удалось создать pipeline layout\n";
		vkDestroyShaderModule(context.device, vertex_module, nullptr);
		vkDestroyShaderModule(context.device, fragment_module, nullptr);
		return false;
	}
	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic,
		.layout = app.pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};
	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline_info, nullptr, &app.pipeline);
	vkDestroyShaderModule(context.device, vertex_module, nullptr);
	vkDestroyShaderModule(context.device, fragment_module, nullptr);
	if (result != VK_SUCCESS) {
		std::cerr << "Не удалось создать graphics pipeline: " << static_cast<int>(result) << '\n';
		return false;
	}
	return true;
}

void updateGlobalUniforms() {
	const auto& context = graphics::internal::context;
	const float aspect = context.swapchain_extent.height == 0 ? 1.0f :
		static_cast<float>(context.swapchain_extent.width) /
		static_cast<float>(context.swapchain_extent.height);
	GlobalUniforms uniforms{};
	uniforms.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 7.2f), glm::vec3(0.0f),
	                           glm::vec3(0.0f, 1.0f, 0.0f));
	if (app.orthographic) {
		const float half_height = 3.05f;
		uniforms.projection = glm::ortho(-half_height * aspect, half_height * aspect,
		                                -half_height, half_height, 0.1f, 30.0f);
	} else {
		uniforms.projection = glm::perspective(glm::radians(48.0f), aspect,
		                                       0.1f, 30.0f);
	}
	uniforms.projection[1][1] *= -1.0f;
	writeUniform(app.global_uniforms, &uniforms, sizeof(uniforms));
}

void updateObjectUniforms() {
	for (Object& object : app.objects) {
		const float angle = app.animation_time + object.phase;
		const glm::vec3 animated_offset = {
			app.orbit_radius * std::cos(angle),
			app.vertical_amplitude * std::sin(2.0f * angle),
			app.orbit_radius * std::sin(angle),
		};
		const glm::vec3 animated_rotation = {
			0.22f * std::sin(angle * 0.7f),
			angle,
			0.18f * std::cos(angle * 0.9f),
		};
		glm::mat4 model = glm::translate(glm::mat4(1.0f), object.position + animated_offset);
		model = glm::rotate(model, glm::radians(object.rotation_degrees.x) + animated_rotation.x,
		                    glm::vec3(1.0f, 0.0f, 0.0f));
		model = glm::rotate(model, glm::radians(object.rotation_degrees.y) + animated_rotation.y,
		                    glm::vec3(0.0f, 1.0f, 0.0f));
		model = glm::rotate(model, glm::radians(object.rotation_degrees.z) + animated_rotation.z,
		                    glm::vec3(0.0f, 0.0f, 1.0f));
		model = glm::scale(model, object.scale);
		const ObjectUniforms uniforms = {
			.model = model,
			.tint = glm::vec4(object.tint, 1.0f),
		};
		writeUniform(object.uniforms, &uniforms, sizeof(uniforms));
	}
}

void drawControls() {
	ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImVec2(345.0f, 355.0f), ImGuiCond_Always);
	ImGui::Begin("Lab 1 - Variant 11", nullptr, ImGuiWindowFlags_NoResize);
	ImGui::Text("Regular icosahedron");
	ImGui::Checkbox("Orthographic projection", &app.orthographic);
	ImGui::Checkbox("Play animation", &app.animation_playing);
	ImGui::SliderFloat("Speed", &app.animation_speed, 0.0f, 2.5f, "%.2f");
	ImGui::SliderFloat("Radius", &app.orbit_radius, 0.0f, 0.8f, "%.2f");
	ImGui::SliderFloat("Height", &app.vertical_amplitude, 0.0f, 0.6f, "%.2f");
	ImGui::Separator();
	const char* object_names[] = { "Icosahedron 1", "Icosahedron 2" };
	int selected = static_cast<int>(app.selected_object);
	if (ImGui::Combo("Object", &selected, object_names, object_count)) {
		app.selected_object = static_cast<uint32_t>(selected);
	}
	Object& object = app.objects[app.selected_object];
	ImGui::DragFloat3("Position", &object.position.x, 0.02f, -4.0f, 4.0f, "%.2f");
	ImGui::DragFloat3("Rotation", &object.rotation_degrees.x, 1.0f,
	                  -180.0f, 180.0f, "%.0f");
	ImGui::DragFloat3("Scale", &object.scale.x, 0.01f, 0.1f, 2.0f, "%.2f");
	ImGui::ColorEdit3("Color", &object.tint.x);
	ImGui::End();
}

} // namespace

bool initialize() {
	app.objects = {{
		{ .position = {-1.25f, 0.0f, 0.0f}, .phase = 0.0f },
		{ .position = {1.25f, 0.0f, -0.5f}, .phase = pi },
	}};
	if (!createIcosahedronMesh() || !createDescriptorLayouts() ||
	    !createDescriptors() || !createPipeline()) {
		destroyResources();
		return false;
	}
	updateGlobalUniforms();
	updateObjectUniforms();
	return true;
}

void shutdown() {
	destroyResources();
}

void update(double time) {
	if (app.has_previous_update_time && app.animation_playing) {
		const double delta = std::clamp(time - app.previous_update_time, 0.0, 0.1);
		app.animation_time += static_cast<float>(delta) * app.animation_speed;
	}
	app.previous_update_time = time;
	app.has_previous_update_time = true;

	drawControls();
}

void render(const graphics::internal::FrameData& fd) {
	if (fd.command_buffer == VK_NULL_HANDLE || fd.framebuffer == VK_NULL_HANDLE ||
	    app.pipeline == VK_NULL_HANDLE) {
		return;
	}
	// prepare() has waited for the previous frame fence: mapped UBOs are now safe to write.
	updateGlobalUniforms();
	updateObjectUniforms();
	vkResetCommandBuffer(fd.command_buffer, 0);
	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	if (vkBeginCommandBuffer(fd.command_buffer, &begin) != VK_SUCCESS) {
		std::cerr << "Не удалось начать запись command buffer\n";
		return;
	}
	const VkClearValue clear_values[] = {
		{ .color = { .float32 = { 0.025f, 0.035f, 0.065f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};
	const VkRenderPassBeginInfo render_pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = graphics::internal::context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = graphics::internal::context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(fd.command_buffer, &render_pass, VK_SUBPASS_CONTENTS_INLINE);
	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = static_cast<float>(graphics::internal::context.swapchain_extent.width),
		.height = static_cast<float>(graphics::internal::context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor = {
		.offset = { 0, 0 },
		.extent = graphics::internal::context.swapchain_extent,
	};
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, app.pipeline);
	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &app.vertex_buffer.handle,
	                      &vertex_offset);
	vkCmdBindIndexBuffer(fd.command_buffer, app.index_buffer.handle, 0, VK_INDEX_TYPE_UINT32);
	for (Object& object : app.objects) {
		const VkDescriptorSet descriptor_sets[] = { app.global_set, object.descriptor_set };
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        app.pipeline_layout, 0, 2, descriptor_sets, 0, nullptr);
		vkCmdDrawIndexed(fd.command_buffer, app.index_count, 1, 0, 0, 0);
	}
	vkCmdEndRenderPass(fd.command_buffer);
	if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
		std::cerr << "Не удалось завершить запись command buffer\n";
	}
}

} // namespace application
