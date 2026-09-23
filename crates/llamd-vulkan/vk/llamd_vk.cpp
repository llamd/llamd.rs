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
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSetLayout dlayout = VK_NULL_HANDLE;
    VkPipelineLayout playout = VK_NULL_HANDLE;
    VkPipeline matvec = VK_NULL_HANDLE;
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

    const char *exts[] = { VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME };
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
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

    VkDescriptorSetLayoutBinding binds[5]{};
    for (int i = 0; i < 5; ++i) {
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 5;
    dli.pBindings = binds;
    // PUSH descriptors would avoid a pool entirely, but the extension is not
    // guaranteed; a pool with a generous set count is portable.
    if (vkCreateDescriptorSetLayout(g.dev, &dli, nullptr, &g.dlayout) != VK_SUCCESS) return -1;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 8 * sizeof(uint32_t);
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &g.dlayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(g.dev, &pli, nullptr, &g.playout) != VK_SUCCESS) return -1;

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 5 * 256;
    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = 256;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(g.dev, &dpi, nullptr, &g.dpool) != VK_SUCCESS) return -1;

    g.matvec = make_pipeline(SPV_Q6K_MATVEC, sizeof(SPV_Q6K_MATVEC) / sizeof(uint32_t));
    if (g.matvec == VK_NULL_HANDLE) return -1;

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
    if (!g.ok || !h) return -1;
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
    if (!g.ok || !h) return -1;
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

// Q6_K matvec. `off` is the four plane offsets in BYTES; the shader wants them
// in uints, converted here so the caller uses the same PlanarQ6K layout the
// ROCm backend does.
int llamd_vk_matvec(void *w, const uint64_t *off, void *qb, void *sb, void *ad,
                    void *out, int blocks, int rows, int nseq, int iters)
{
    if (!g.ok) return -1;
    Buf *bw = (Buf *)w, *bq = (Buf *)qb, *bs = (Buf *)sb, *ba = (Buf *)ad, *bo = (Buf *)out;

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = g.dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &g.dlayout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(g.dev, &ai, &set) != VK_SUCCESS) return -1;

    VkBuffer bufs[5] = {bw->buf, bq->buf, bs->buf, ba->buf, bo->buf};
    VkDescriptorBufferInfo bi[5]{};
    VkWriteDescriptorSet wr[5]{};
    for (int i = 0; i < 5; ++i) {
        bi[i].buffer = bufs[i];
        bi[i].offset = 0;
        bi[i].range = VK_WHOLE_SIZE;
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = set;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(g.dev, 5, wr, 0, nullptr);

    uint32_t pc[8];
    pc[0] = (uint32_t)(off[0] / 4);
    pc[1] = (uint32_t)(off[1] / 4);
    pc[2] = (uint32_t)(off[2] / 4);
    pc[3] = (uint32_t)(off[3] / 4);
    pc[4] = (uint32_t)blocks;
    pc[5] = (uint32_t)rows;
    pc[6] = (uint32_t)nseq;
    pc[7] = 0;

    // One subgroup per row; 256 invocations per workgroup.
    const uint32_t subgroups_per_wg = 256u / (g.subgroup_size ? g.subgroup_size : 64u);
    const uint32_t groups = ((uint32_t)rows + subgroups_per_wg - 1) / subgroups_per_wg;

    if (!begin_cmd()) return -1;
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.matvec);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.playout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(g.cmd, g.playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    for (int i = 0; i < (iters > 0 ? iters : 1); ++i) {
        vkCmdDispatch(g.cmd, groups, 1, 1);
        if (i + 1 < iters) {
            // Back-to-back timing runs must not overlap, or the measurement is
            // of the queue rather than the kernel.
            VkMemoryBarrier mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(g.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        }
    }
    if (vkEndCommandBuffer(g.cmd) != VK_SUCCESS) return -1;
    const bool okk = submit_and_wait();
    vkFreeDescriptorSets(g.dev, g.dpool, 1, &set);
    return okk ? 0 : -1;
}

} // extern "C"
