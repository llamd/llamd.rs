// Vulkan compute host for LLAMD.
//
// WHY C++ AND NOT A RUST VULKAN CRATE. This workspace's dependency policy is
// explicit: "Every dependency in the numerical path is a thing that must be
// trusted or verified, so the numerical path has none." llamd-rocm already puts
// its device code in C++ behind a small C API with a thin safe Rust wrapper;
// this mirrors that exactly, so the two backends are comparable in structure as
// well as in arithmetic, and adds no crate to the numerical path.
//
// WHY VULKAN IS THE PRIMARY BACKEND. It is the vendor-neutral compute API this
// card exposes natively through RADV, and on this device it offers everything
// the ROCm path uses: integerDotProduct4x8BitPackedUnsignedAccelerated (the
// udot4 equivalent, in hardware), shaderInt8, 8-bit storage, subgroup
// arithmetic, subgroup size control, and a 31.75 GiB device-local heap. ROCm
// stays fully supported as the benchmarking reference.
//
// SCOPE OF THIS FILE: device setup, memory, pipelines and dispatch. The
// arithmetic lives in shaders/, and is the same arithmetic as the HIP kernels
// so that a comparison between backends measures the backend.

#include <vulkan/vulkan.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <vector>

namespace {

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

struct Ctx {
    bool ok = false;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    PFN_vkCmdPushDescriptorSetKHR push_desc = nullptr;
    // There is ONE command buffer. An upload or download between begin() and
    // end_and_wait() would reset it and silently discard a half-recorded token,
    // producing plausible wrong output rather than an error — so it is refused.
    bool recording = false;
    VkDescriptorSetLayout dlayout = VK_NULL_HANDLE;
    VkPipelineLayout playout = VK_NULL_HANDLE;
    // One pipeline per shader, indexed by the build order in build.rs.
    std::vector<VkPipeline> pipes;
    VkPhysicalDeviceMemoryProperties memprops{};
    char name[256] = {0};
    char driver[256] = {0};
    uint32_t subgroup_size = 0;
    uint64_t heap_bytes = 0;
    // Staging buffer for host<->device copies, grown on demand. A persistent
    // one matters for the same reason it did on the ROCm side: per-call
    // allocation dominated everything.
    Buf staging;
};

Ctx g;

uint32_t find_mem(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g.memprops.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) &&
            (g.memprops.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

bool make_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                 VkMemoryPropertyFlags props, Buf *out) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(g.dev, &bi, nullptr, &out->buf) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g.dev, out->buf, &req);
    const uint32_t type = find_mem(req.memoryTypeBits, props);
    if (type == UINT32_MAX) {
        vkDestroyBuffer(g.dev, out->buf, nullptr);
        out->buf = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(g.dev, &ai, nullptr, &out->mem) != VK_SUCCESS) {
        vkDestroyBuffer(g.dev, out->buf, nullptr);
        out->buf = VK_NULL_HANDLE;
        return false;
    }
    if (vkBindBufferMemory(g.dev, out->buf, out->mem, 0) != VK_SUCCESS) return false;
    out->size = size;
    return true;
}

void drop_buffer(Buf *b) {
    if (b->buf) vkDestroyBuffer(g.dev, b->buf, nullptr);
    if (b->mem) vkFreeMemory(g.dev, b->mem, nullptr);
    b->buf = VK_NULL_HANDLE;
    b->mem = VK_NULL_HANDLE;
    b->size = 0;
}

bool ensure_staging(VkDeviceSize want) {
    if (g.staging.size >= want) return true;
    drop_buffer(&g.staging);
    return make_buffer(want,
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       &g.staging);
}

bool submit_and_wait() {
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g.cmd;
    vkResetFences(g.dev, 1, &g.fence);
    if (vkQueueSubmit(g.queue, 1, &si, g.fence) != VK_SUCCESS) return false;
    return vkWaitForFences(g.dev, 1, &g.fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
}

bool begin_cmd() {
    vkResetCommandBuffer(g.cmd, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    return vkBeginCommandBuffer(g.cmd, &bi) == VK_SUCCESS;
}

// The SPIR-V, embedded by build.rs.
#include "shaders_spv.h"

VkPipeline make_pipeline(const uint32_t *spv, size_t words) {
    VkShaderModuleCreateInfo mi{};
    mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    mi.codeSize = words * sizeof(uint32_t);
    mi.pCode = spv;
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(g.dev, &mi, nullptr, &mod) != VK_SUCCESS) return VK_NULL_HANDLE;

    VkPipelineShaderStageCreateInfo st{};
    st.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    st.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    st.module = mod;
    st.pName = "main";

    VkComputePipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage = st;
    ci.layout = g.playout;
    VkPipeline p = VK_NULL_HANDLE;
    const VkResult r = vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p);
    vkDestroyShaderModule(g.dev, mod, nullptr);
    return (r == VK_SUCCESS) ? p : VK_NULL_HANDLE;
}

} // namespace

extern "C" {

int llamd_vk_init() {
    if (g.ok) return 0;

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "llamd";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &g.instance) != VK_SUCCESS) return -1;

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g.instance, &n, nullptr);
    if (n == 0) return -1;
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(g.instance, &n, devs.data());

    // PREFER A DISCRETE GPU. This machine also exposes llvmpipe, a software
    // device that reports Vulkan support and would silently make every
    // measurement meaningless.
    VkPhysicalDevice pick = VK_NULL_HANDLE;
    for (auto d : devs) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pick = d; break; }
    }
    if (pick == VK_NULL_HANDLE) return -1;
    g.phys = pick;

    VkPhysicalDeviceSubgroupProperties sg{};
    sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &sg;
    vkGetPhysicalDeviceProperties2(g.phys, &p2);
    snprintf(g.name, sizeof(g.name), "%s", p2.properties.deviceName);
    g.subgroup_size = sg.subgroupSize;

    VkPhysicalDeviceDriverProperties dp{};
    dp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceProperties2 p2b{};
    p2b.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2b.pNext = &dp;
    vkGetPhysicalDeviceProperties2(g.phys, &p2b);
    snprintf(g.driver, sizeof(g.driver), "%s %s", dp.driverName, dp.driverInfo);

    vkGetPhysicalDeviceMemoryProperties(g.phys, &g.memprops);
    for (uint32_t i = 0; i < g.memprops.memoryHeapCount; ++i) {
        if (g.memprops.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            if (g.memprops.memoryHeaps[i].size > g.heap_bytes)
                g.heap_bytes = g.memprops.memoryHeaps[i].size;
        }
    }

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &qn, qfs.data());
    // A compute-only family if there is one: it avoids sharing with graphics
    // work and is what the card exposes for exactly this purpose.
    uint32_t chosen = UINT32_MAX;
    for (uint32_t i = 0; i < qn; ++i) {
        const auto f = qfs[i].queueFlags;
        if ((f & VK_QUEUE_COMPUTE_BIT) && !(f & VK_QUEUE_GRAPHICS_BIT)) { chosen = i; break; }
    }
    if (chosen == UINT32_MAX) {
        for (uint32_t i = 0; i < qn; ++i)
            if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { chosen = i; break; }
    }
    if (chosen == UINT32_MAX) return -1;
    g.qfam = chosen;

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = g.qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    // The features the shaders require. Requested explicitly so a device that
    // lacks them fails here, legibly, rather than miscompiling a shader.
    VkPhysicalDeviceShaderIntegerDotProductFeatures dot{};
    dot.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    dot.shaderIntegerDotProduct = VK_TRUE;
    VkPhysicalDeviceVulkan12Features v12{};
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.shaderInt8 = VK_TRUE;
    v12.storageBuffer8BitAccess = VK_TRUE;
    v12.pNext = &dot;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &v12;

    const char *exts[] = {
        VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
    };
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 2;
    dci.ppEnabledExtensionNames = exts;
    if (vkCreateDevice(g.phys, &dci, nullptr, &g.dev) != VK_SUCCESS) return -1;
    vkGetDeviceQueue(g.dev, g.qfam, 0, &g.queue);

    VkCommandPoolCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = g.qfam;
    if (vkCreateCommandPool(g.dev, &cpi, nullptr, &g.pool) != VK_SUCCESS) return -1;

    VkCommandBufferAllocateInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = g.pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g.dev, &cbi, &g.cmd) != VK_SUCCESS) return -1;

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(g.dev, &fi, nullptr, &g.fence) != VK_SUCCESS) return -1;

    VkDescriptorSetLayoutBinding binds[8]{};
    for (int i = 0; i < 8; ++i) {
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 8;
    dli.pBindings = binds;
    // PUSH DESCRIPTORS: the buffer handles travel inside the command buffer, so
    // there is no pool, no set allocation and nothing to recycle between
    // dispatches. A token is ~15 dispatches per layer and allocating a
    // descriptor set for each would dominate.
    dli.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    if (vkCreateDescriptorSetLayout(g.dev, &dli, nullptr, &g.dlayout) != VK_SUCCESS) return -1;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 16 * sizeof(uint32_t);   // 64 bytes; the device guarantees 128
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g.dlayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(g.dev, &pli, nullptr, &g.playout) != VK_SUCCESS) return -1;

    g.push_desc = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(
        g.dev, "vkCmdPushDescriptorSetKHR");
    if (!g.push_desc) return -1;

    const size_t n_pipes = sizeof(SPV_TABLE) / sizeof(SPV_TABLE[0]);
    g.pipes.resize(n_pipes, VK_NULL_HANDLE);
    for (size_t i = 0; i < n_pipes; ++i) {
        g.pipes[i] = make_pipeline(SPV_TABLE[i].code, SPV_TABLE[i].words);
        if (g.pipes[i] == VK_NULL_HANDLE) {
            fprintf(stderr, "llamd-vulkan: pipeline %s failed to create\n", SPV_TABLE[i].name);
            return -1;
        }
    }

    g.ok = true;
    return 0;
}

int llamd_vk_available() { return g.ok ? 1 : 0; }

int llamd_vk_device_info(char *name, int name_len, char *driver, int driver_len,
                         uint64_t *heap, uint32_t *subgroup) {
    if (!g.ok) return -1;
    snprintf(name, name_len, "%s", g.name);
    snprintf(driver, driver_len, "%s", g.driver);
    *heap = g.heap_bytes;
    *subgroup = g.subgroup_size;
    return 0;
}

void *llamd_vk_alloc(uint64_t bytes) {
    if (!g.ok || bytes == 0) return nullptr;
    Buf *b = new Buf();
    if (!make_buffer(bytes,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b)) {
        delete b;
        return nullptr;
    }
    return b;
}

void llamd_vk_free(void *h) {
    if (!h) return;
    Buf *b = (Buf *)h;
    drop_buffer(b);
    delete b;
}

int llamd_vk_upload(void *h, const void *src, uint64_t bytes) {
    if (!g.ok || !h || g.recording) return -1;
    Buf *b = (Buf *)h;
    if (bytes > b->size) return -1;
    if (!ensure_staging(bytes)) return -1;
    void *map = nullptr;
    if (vkMapMemory(g.dev, g.staging.mem, 0, bytes, 0, &map) != VK_SUCCESS) return -1;
    memcpy(map, src, bytes);
    vkUnmapMemory(g.dev, g.staging.mem);
    if (!begin_cmd()) return -1;
    VkBufferCopy cp{0, 0, bytes};
    vkCmdCopyBuffer(g.cmd, g.staging.buf, b->buf, 1, &cp);
    if (vkEndCommandBuffer(g.cmd) != VK_SUCCESS) return -1;
    return submit_and_wait() ? 0 : -1;
}

int llamd_vk_download(void *h, void *dst, uint64_t bytes) {
    if (!g.ok || !h || g.recording) return -1;
    Buf *b = (Buf *)h;
    if (bytes > b->size) return -1;
    if (!ensure_staging(bytes)) return -1;
    if (!begin_cmd()) return -1;
    VkBufferCopy cp{0, 0, bytes};
    vkCmdCopyBuffer(g.cmd, b->buf, g.staging.buf, 1, &cp);
    if (vkEndCommandBuffer(g.cmd) != VK_SUCCESS) return -1;
    if (!submit_and_wait()) return -1;
    void *map = nullptr;
    if (vkMapMemory(g.dev, g.staging.mem, 0, bytes, 0, &map) != VK_SUCCESS) return -1;
    memcpy(dst, map, bytes);
    vkUnmapMemory(g.dev, g.staging.mem);
    return 0;
}

// ---------------------------------------------------------------------------
// RECORD, DISPATCH, SUBMIT ONCE.
//
// A token is roughly fifteen dispatches per layer. Recording them all into one
// command buffer and waiting once is the Vulkan equivalent of the ROCm path's
// single hipDeviceSynchronize, and for the same reason: the alternative is a
// stall per operation, which on the ROCm side cost 0.12 ms per matmul and put a
// hard ~6 tok/s ceiling on the engine regardless of kernel speed.
//
// Buffer handles travel inside the command buffer via push descriptors, so there
// is no descriptor pool and nothing to recycle between dispatches.
//
// A COMPUTE-TO-COMPUTE BARRIER FOLLOWS EVERY DISPATCH. Almost every operation
// in a token reads what the previous one wrote; Vulkan does not order dispatches
// implicitly the way a HIP stream does, and omitting the barrier produces
// plausible, wrong output rather than an error.
int llamd_vk_pipeline_count() {
    return (int)(sizeof(SPV_TABLE) / sizeof(SPV_TABLE[0]));
}

const char *llamd_vk_pipeline_name(int i) {
    if (i < 0 || i >= llamd_vk_pipeline_count()) return "";
    return SPV_TABLE[i].name;
}

int llamd_vk_begin() {
    if (!g.ok || g.recording) return -1;
    if (!begin_cmd()) return -1;
    g.recording = true;
    return 0;
}

int llamd_vk_dispatch(int pipe, void *const *bufs, int nbuf,
                      const uint32_t *push, uint32_t gx, uint32_t gy)
{
    if (!g.ok || !g.recording) return -1;
    if (pipe < 0 || pipe >= llamd_vk_pipeline_count()) return -1;
    if (nbuf < 0 || nbuf > 8) return -1;
    if (gx == 0 || gy == 0) return 0;          // nothing to do, not an error

    VkDescriptorBufferInfo bi[8]{};
    VkWriteDescriptorSet wr[8]{};
    for (int i = 0; i < nbuf; ++i) {
        Buf *b = (Buf *)bufs[i];
        if (!b || b->buf == VK_NULL_HANDLE) return -1;
        bi[i].buffer = b->buf;
        bi[i].offset = 0;
        bi[i].range = VK_WHOLE_SIZE;
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &bi[i];
    }

    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipes[(size_t)pipe]);
    g.push_desc(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.playout, 0, (uint32_t)nbuf, wr);
    vkCmdPushConstants(g.cmd, g.playout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       16 * sizeof(uint32_t), push);
    vkCmdDispatch(g.cmd, gx, gy, 1);

    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr,
                         0, nullptr);
    return 0;
}

int llamd_vk_end_and_wait() {
    if (!g.ok || !g.recording) return -1;
    g.recording = false;
    if (vkEndCommandBuffer(g.cmd) != VK_SUCCESS) return -1;
    return submit_and_wait() ? 0 : -1;
}

uint32_t llamd_vk_subgroup_size() { return g.subgroup_size; }

// Convenience wrapper for the standalone matvec validation and benchmark: one
// begin / N dispatches / one wait.
int llamd_vk_matvec(void *w, const uint64_t *off, void *qb, void *sb, void *ad,
                    void *out, int blocks, int rows, int nseq, int iters)
{
    if (!g.ok) return -1;
    uint32_t pc[16] = {0};
    pc[0] = (uint32_t)(off[0] / 4);
    pc[1] = (uint32_t)(off[1] / 4);
    pc[2] = (uint32_t)(off[2] / 4);
    pc[3] = (uint32_t)(off[3] / 4);
    pc[4] = (uint32_t)blocks;
    pc[5] = (uint32_t)rows;
    pc[6] = (uint32_t)nseq;

    void *bufs[5] = {w, qb, sb, ad, out};
    const uint32_t sg = g.subgroup_size ? g.subgroup_size : 64u;
    const uint32_t per_wg = 256u / sg;                 // subgroups, one row each
    const uint32_t groups = ((uint32_t)rows + per_wg - 1) / per_wg;

    if (llamd_vk_begin() != 0) return -1;
    for (int i = 0; i < (iters > 0 ? iters : 1); ++i) {
        if (llamd_vk_dispatch(0, bufs, 5, pc, groups, 1) != 0) return -1;
    }
    return llamd_vk_end_and_wait();
}

} // extern "C"
