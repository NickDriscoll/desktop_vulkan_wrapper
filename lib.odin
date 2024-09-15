package desktop_vulkan_wrapper

import "core:container/queue"
import "core:fmt"
import "core:os"
import "core:log"
import "core:math"
import win32 "core:sys/windows"
import "vendor:sdl2"
import vk "vendor:vulkan"

import "odin-vma/vma"
import hm "handlemap"

// @HACK: This is a struct in Vulkan that the Odin bindings are missing at the moment
// This really should be replaced in the future
vkPhysicalDeviceMaintenance5FeaturesKHR :: struct {
    sType: vk.StructureType,
    pNext: rawptr,
    maintenance5: b32
}
VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR :: 1000470000

MAXIMUM_BINDLESS_IMAGES :: 1024 * 1024
TOTAL_SAMPLERS :: 2
IMAGES_DESCRIPTOR_BINDING :: 0
SAMPLERS_DESCRIPTOR_BINDING :: 1
PUSH_CONSTANTS_SIZE :: 128

IDENTITY_COMPONENT_SWIZZLE :: vk.ComponentMapping {
    r = .R,
    g = .G,
    b = .B,
    a = .A,
}

float2 :: [2]f32
float3 :: [3]f32
float4 :: [4]f32
int2 :: [2]i32
uint2 :: [2]u32

Queue_Family :: enum {
    Graphics,
    Compute,
    Transfer
}

Semaphore_Info :: struct {
    type: vk.SemaphoreType,
    init_value: u64,
}

Semaphore_Op :: struct {
    semaphore: Semaphore_Handle,
    value: u64
}

Sync_Info :: struct {
    wait_ops: [dynamic]Semaphore_Op,
    signal_ops: [dynamic]Semaphore_Op,
}

delete_sync_info :: proc(s: ^Sync_Info) {
    delete(s.wait_ops)
    delete(s.signal_ops)
}

Buffer :: struct {
    buffer: vk.Buffer,
    allocation: vma.Allocation,
    alloc_info: vma.Allocation_Info
}

Buffer_Delete :: struct {
    death_frame: u64,
    buffer: vk.Buffer,
    allocation: vma.Allocation
}

Image :: struct {
    image: vk.Image,
    image_view: vk.ImageView,
    allocation: vma.Allocation
}

// Distinct handle types for each Handle_Map in the Graphics_Device
Buffer_Handle :: distinct hm.Handle
Image_Handle :: distinct hm.Handle
Semaphore_Handle :: distinct hm.Handle

// Megastruct holding basically all Vulkan-specific state
Graphics_Device :: struct {
    // Basic Vulkan state that every app definitely needs
    instance: vk.Instance,
    physical_device: vk.PhysicalDevice,
    device: vk.Device,
    pipeline_cache: vk.PipelineCache,
    alloc_callbacks: ^vk.AllocationCallbacks,
    allocator: vma.Allocator,
    //vma_alloc_callbacks: ^vma.Allcation_ca,
    frames_in_flight: u32,
    frame_count: u64,
    
    // Objects required to support windowing
    // Basically every app will use these, but maybe
    // these could be factored out
    surface: vk.SurfaceKHR,
    swapchain: vk.SwapchainKHR,
    swapchain_images: [dynamic]Image_Handle,
    acquire_semaphores: [dynamic]Semaphore_Handle,
    present_semaphores: [dynamic]Semaphore_Handle,
    
    
    // The Vulkan queues that the device will submit on
    // May be aliases of each other if e.g. the GPU doesn't have
    // an async compute queue
    gfx_queue_family: u32,
    compute_queue_family: u32,
    transfer_queue_family: u32,
    gfx_queue: vk.Queue,
    compute_queue: vk.Queue,
    transfer_queue: vk.Queue,
    
    // Command buffer related state
    gfx_command_pool: vk.CommandPool,
    compute_command_pool: vk.CommandPool,
    transfer_command_pool: vk.CommandPool,
    gfx_command_buffers: [dynamic]vk.CommandBuffer,
    compute_command_buffers: [dynamic]vk.CommandBuffer,
    transfer_command_buffers: [dynamic]vk.CommandBuffer,
    next_gfx_command_buffer: u32,

    // All users of this Vulkan wrapper will access
    // buffers via their device addresses and
    // images through this global descriptor set
    // i.e. all bindless all the time, baby
    immutable_samplers: [TOTAL_SAMPLERS]vk.Sampler,
    descriptor_set_layout: vk.DescriptorSetLayout,
    descriptor_pool: vk.DescriptorPool,
    descriptor_set: vk.DescriptorSet,


    // Pipeline layout used for all pipelines
    pipeline_layout: vk.PipelineLayout,
    
    // Handle_Maps of all Vulkan objects
    buffers: hm.Handle_Map(Buffer),
    images: hm.Handle_Map(Image),
    semaphores: hm.Handle_Map(vk.Semaphore),
    pipelines: hm.Handle_Map(vk.Pipeline),

    // Deletion queues for Buffers and Images
    buffer_deletes: queue.Queue(Buffer_Delete)
    
    
}

API_Version :: enum {
    Vulkan12,
    Vulkan13
}

Init_Parameters :: struct {
    // Vulkan instance creation parameters
    app_name: cstring,
    app_version: u32,
    engine_name: cstring,
    engine_version: u32,
    api_version: API_Version,
    
    allocation_callbacks: ^vk.AllocationCallbacks,
    vk_get_instance_proc_addr: rawptr,
    
    frames_in_flight: u32,      // Maximum number of command buffers active at once
    
    
    window_support: bool        // Will this device need to draw to window surface swapchains?

}

init_vulkan :: proc(using params: ^Init_Parameters) -> Graphics_Device {
    assert(frames_in_flight > 0)
    
    log.log(.Info, "Initializing Vulkan instance and device")
    
    if vk_get_instance_proc_addr == nil {
        log.fatal("Init_Paramenters.vk_get_instance_proc_addr was nil!")
    }
    vk.load_proc_addresses_global(vk_get_instance_proc_addr)
    
    // Create Vulkan instance
    // @TODO: Look into vkEnumerateInstanceVersion()
    inst: vk.Instance
    api_version_int: u32
    {
        switch api_version {
            case .Vulkan12:
                log.info("Selected Vulkan 1.2")
                api_version_int = vk.API_VERSION_1_2
            case .Vulkan13:
                log.info("Selected Vulkan 1.3")
                api_version_int = vk.API_VERSION_1_3
        }
            
        
        // Instead of forcing the caller to explicitly provide
        // the extensions they want to enable, I want to provide high-level
        // idioms that cover many extensions in the same logical category
        extensions: [dynamic]cstring
        defer delete(extensions)
        if window_support {
            append(&extensions, vk.KHR_SURFACE_EXTENSION_NAME)
            when ODIN_OS == .Windows {
                append(&extensions, vk.KHR_WIN32_SURFACE_EXTENSION_NAME)
            }
            when ODIN_OS == .Linux {
                // @TODO: why is there no vk.KHR_XLIB_SURFACE_EXTENSION_NAME?
                append(&extensions, "VK_KHR_xlib_surface")
            }
        }
        
        app_info := vk.ApplicationInfo {
            sType = .APPLICATION_INFO,
            pNext = nil,
            pApplicationName = app_name,
            applicationVersion = app_version,
            pEngineName = engine_name,
            engineVersion = engine_version,
            apiVersion = api_version_int
        }
        create_info := vk.InstanceCreateInfo {
            sType = .INSTANCE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            pApplicationInfo = &app_info,
            enabledLayerCount = 0,
            ppEnabledLayerNames = nil,
            enabledExtensionCount = u32(len(extensions)),
            ppEnabledExtensionNames = raw_data(extensions)
        }
        
        if vk.CreateInstance(&create_info, allocation_callbacks, &inst) != .SUCCESS {
            log.fatal("Instance creation failed.")
        }
    }

    // Load instance-level procedures
    vk.load_proc_addresses_instance(inst)

    // Create Vulkan device
    phys_device: vk.PhysicalDevice
    device: vk.Device
    gfx_queue_family: u32
    compute_queue_family: u32
    transfer_queue_family: u32
    {
        phys_device_count : u32 = 0
        vk.EnumeratePhysicalDevices(inst, &phys_device_count, nil)
        
        phys_devices: [dynamic]vk.PhysicalDevice
        resize(&phys_devices, int(phys_device_count))
        defer delete(phys_devices)
        vk.EnumeratePhysicalDevices(inst, &phys_device_count, raw_data(phys_devices))
        
        // Select the physical device to use
        // @NOTE: We only support using a single physical device at once
        features: vk.PhysicalDeviceFeatures2
        for pd in phys_devices {
            // Query this physical device's properties
            vk12_props: vk.PhysicalDeviceVulkan12Properties
            vk12_props.sType = .PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES
            props: vk.PhysicalDeviceProperties2
            props.sType = .PHYSICAL_DEVICE_PROPERTIES_2
            props.pNext = &vk12_props
            vk.GetPhysicalDeviceProperties2(pd, &props)

            // @TODO: Do something more sophisticated than picking the first DISCRETE_GPU
            if props.properties.deviceType == .DISCRETE_GPU {
                // Check physical device features
                dynamic_rendering_features: vk.PhysicalDeviceDynamicRenderingFeatures
                timeline_features: vk.PhysicalDeviceTimelineSemaphoreFeatures
                sync2_features: vk.PhysicalDeviceSynchronization2Features
                bda_features: vk.PhysicalDeviceBufferDeviceAddressFeatures
                maint5_features: vkPhysicalDeviceMaintenance5FeaturesKHR

                dynamic_rendering_features.sType = .PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES
                timeline_features.sType = .PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES
                sync2_features.sType = .PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES
                bda_features.sType = .PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES
                maint5_features.sType = vk.StructureType(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR)
                features.sType = .PHYSICAL_DEVICE_FEATURES_2

                dynamic_rendering_features.pNext = &maint5_features
                timeline_features.pNext = &dynamic_rendering_features
                sync2_features.pNext = &timeline_features
                bda_features.pNext = &sync2_features
                features.pNext = &bda_features
                vk.GetPhysicalDeviceFeatures2(pd, &features)
                log.debugf("%#v", features)
                log.debugf("%#v", dynamic_rendering_features)
                log.debugf("%#v", timeline_features)
                log.debugf("%#v", bda_features)
                log.debugf("%#v", maint5_features)

                has_right_features :=
                    dynamic_rendering_features.dynamicRendering &&
                    maint5_features.maintenance5 &&
                    bda_features.bufferDeviceAddress && 
                    sync2_features.synchronization2 &&
                    timeline_features.timelineSemaphore
                if has_right_features {
                    phys_device = pd
                    log.infof("Chosen GPU: %s", string(props.properties.deviceName[:]))
                    break
                }
            }
        }

        assert(phys_device != nil, "Didn't find vkPhysicalDevice")

        // Query the physical device's queue family properties
        queue_family_count : u32 = 0
        vk.GetPhysicalDeviceQueueFamilyProperties2(phys_device, &queue_family_count, nil)

        qfps: [dynamic]vk.QueueFamilyProperties2
        resize(&qfps, int(queue_family_count))
        defer delete(qfps)
        for &qfp in qfps {
            qfp.sType = .QUEUE_FAMILY_PROPERTIES_2
        }
        vk.GetPhysicalDeviceQueueFamilyProperties2(phys_device, &queue_family_count, raw_data(qfps))

        //Load all supported device extensions for later querying
        extension_count : u32 = 0
        vk.EnumerateDeviceExtensionProperties(phys_device, nil, &extension_count, nil)

        device_extensions: [dynamic]vk.ExtensionProperties
        defer delete(device_extensions)
        resize(&device_extensions, int(extension_count))
        vk.EnumerateDeviceExtensionProperties(phys_device, nil, &extension_count, raw_data(device_extensions))

        // Query for extension support,
        // namely Sync2 and dynamic rendering support for now
        comp_bytes_to_string :: proc(bytes: []byte, s: string) -> bool {
            return string(bytes[0:len(s)]) == s
        }
        
        necessary_extensions: []string
        switch api_version {
            case .Vulkan12: {
                necessary_extensions = {
                    vk.KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
                    vk.KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
                    vk.KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
                    "VK_KHR_maintenance5"
                }
            }
            case .Vulkan13: {
                necessary_extensions = {
                    "VK_KHR_maintenance5"
                }
            }
        }

        extension_flags: bit_set[0..<4] = {}
        for ext in device_extensions {
            name := ext.extensionName
            for ext_string, i in necessary_extensions {
                if comp_bytes_to_string(name[:], ext_string) {
                    extension_flags += {i}
                    log.debugf("%s verified", name)
                }
            }
        }
        for s, i in necessary_extensions {
            if i not_in extension_flags {
                log.fatalf("Your device does not support %v. Buh bye.", s)
            }
        }

        // Determine available queue family types
        for qfp, i in qfps {
            flags := qfp.queueFamilyProperties.queueFlags
            if vk.QueueFlag.GRAPHICS in flags do gfx_queue_family = u32(i)
        }
        compute_queue_family = gfx_queue_family
        transfer_queue_family = gfx_queue_family
        log.debug("Queue family profile flags...")
        for qfp, i in qfps {
            flags := qfp.queueFamilyProperties.queueFlags
            log.debugf("%#v", qfp.queueFamilyProperties.queueFlags)
            
            if .COMPUTE&.TRANSFER in flags {
                compute_queue_family = u32(i)
                transfer_queue_family = u32(i)
            }
            else if vk.QueueFlag.COMPUTE in flags do compute_queue_family = u32(i)
            else if vk.QueueFlag.TRANSFER in flags do transfer_queue_family = u32(i)
        }

        queue_priority : f32 = 1.0
        queue_count : u32 = 1
        queue_create_infos: [3]vk.DeviceQueueCreateInfo
        queue_create_infos[0] = vk.DeviceQueueCreateInfo {
            sType = .DEVICE_QUEUE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            queueFamilyIndex = gfx_queue_family,
            queueCount = 1,
            pQueuePriorities = &queue_priority
        }
        if compute_queue_family != gfx_queue_family {
            queue_count += 1
            queue_create_infos[1] = vk.DeviceQueueCreateInfo {
                sType = .DEVICE_QUEUE_CREATE_INFO,
                pNext = nil,
                flags = nil,
                queueFamilyIndex = compute_queue_family,
                queueCount = 1,
                pQueuePriorities = &queue_priority
            }
        }
        if transfer_queue_family != compute_queue_family {
            queue_count += 1
            queue_create_infos[2] = vk.DeviceQueueCreateInfo {
                sType = .DEVICE_QUEUE_CREATE_INFO,
                pNext = nil,
                flags = nil,
                queueFamilyIndex = transfer_queue_family,
                queueCount = 1,
                pQueuePriorities = &queue_priority
            }
        }

        // Device extensions
        extensions: [dynamic]cstring
        defer delete(extensions)
        if window_support {
            append(&extensions, vk.KHR_SWAPCHAIN_EXTENSION_NAME)
        }
        if api_version == .Vulkan12 {
            append(&extensions, vk.KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
            append(&extensions, vk.KHR_DYNAMIC_RENDERING_EXTENSION_NAME)
            append(&extensions, vk.KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)
            append(&extensions, "VK_KHR_maintenance5")
        }
        
        // Create logical device
        create_info := vk.DeviceCreateInfo {
            sType = .DEVICE_CREATE_INFO,
            pNext = &features,
            flags = nil,
            queueCreateInfoCount = queue_count,
            pQueueCreateInfos = raw_data(&queue_create_infos),
            enabledExtensionCount = u32(len(extensions)),
            ppEnabledExtensionNames = raw_data(extensions),
            ppEnabledLayerNames = nil,
            pEnabledFeatures = nil
        }
        
        if vk.CreateDevice(phys_device, &create_info, allocation_callbacks, &device) != .SUCCESS {
            log.fatal("Failed to create device.")
        }
    }

    //Load proc addrs that come from the device driver
    vk.load_proc_addresses_device(device)

    // Drivers won't report a function with e.g. a KHR suffix if it has
    // been promoted to core in the requested Vulkan version which is super annoying
    {
        if vk.QueueSubmit2 == nil do vk.QueueSubmit2 = vk.QueueSubmit2KHR
        if vk.QueueSubmit2KHR == nil do vk.QueueSubmit2KHR = vk.QueueSubmit2
        if vk.CmdBeginRendering == nil do vk.CmdBeginRendering = vk.CmdBeginRenderingKHR
        if vk.CmdBeginRenderingKHR == nil do vk.CmdBeginRenderingKHR = vk.CmdBeginRendering
        if vk.CmdEndRendering == nil do vk.CmdEndRendering = vk.CmdEndRenderingKHR
        if vk.CmdEndRenderingKHR == nil do vk.CmdEndRenderingKHR = vk.CmdEndRendering
        if vk.CmdPipelineBarrier2 == nil do vk.CmdPipelineBarrier2 = vk.CmdPipelineBarrier2KHR
        if vk.CmdPipelineBarrier2KHR == nil do vk.CmdPipelineBarrier2KHR = vk.CmdPipelineBarrier2
    }

    // Initialize the Vulkan Memory Allocator
    allocator: vma.Allocator
    {
        // We have to redefine these function aliases in order to
        // ensure VMA is able to find all the functions
        vk.GetBufferMemoryRequirements2KHR = vk.GetBufferMemoryRequirements2
        vk.GetImageMemoryRequirements2KHR = vk.GetImageMemoryRequirements2
        vk.BindBufferMemory2KHR = vk.BindBufferMemory2
        vk.BindImageMemory2KHR = vk.BindImageMemory2
        vk.GetPhysicalDeviceMemoryProperties2KHR = vk.GetPhysicalDeviceMemoryProperties2
        fns := vma.create_vulkan_functions()

        info := vma.Allocator_Create_Info {
            flags = {.Externally_Synchronized,.Buffer_Device_Address},
            physical_device = phys_device,
            device = device,
            preferred_large_heap_block_size = 0,
            allocation_callbacks = allocation_callbacks,
            device_memory_callbacks = nil,
            heap_size_limit = nil,
            vulkan_functions = &fns,
            instance = inst,
            vulkan_api_version = api_version_int,
            type_external_memory_handle_types = nil            
        }
        if vma.create_allocator(&info, &allocator) != .SUCCESS {
            log.fatal("Failed to initialize VMA.")
        }
    }

    // Cache individual queues
    // We only use one queue from each family
    gfx_queue: vk.Queue
    compute_queue: vk.Queue
    transfer_queue: vk.Queue
    {
        vk.GetDeviceQueue(device, gfx_queue_family, 0, &gfx_queue)
        vk.GetDeviceQueue(device, compute_queue_family, 0, &compute_queue)
        vk.GetDeviceQueue(device, transfer_queue_family, 0, &transfer_queue)
    }

    // Create command buffer state
    gfx_command_pool: vk.CommandPool
    compute_command_pool: vk.CommandPool
    transfer_command_pool: vk.CommandPool
    {
        gfx_pool_info := vk.CommandPoolCreateInfo {
            sType = .COMMAND_POOL_CREATE_INFO,
            pNext = nil,
            flags = {vk.CommandPoolCreateFlag.TRANSIENT, vk.CommandPoolCreateFlag.RESET_COMMAND_BUFFER},
            queueFamilyIndex = gfx_queue_family
        }
        if vk.CreateCommandPool(device, &gfx_pool_info, allocation_callbacks, &gfx_command_pool) != .SUCCESS {
            log.fatal("Failed to create gfx command pool")
        }
        
        compute_pool_info := vk.CommandPoolCreateInfo {
            sType = .COMMAND_POOL_CREATE_INFO,
            pNext = nil,
            flags = {vk.CommandPoolCreateFlag.TRANSIENT, vk.CommandPoolCreateFlag.RESET_COMMAND_BUFFER},
            queueFamilyIndex = compute_queue_family
        }
        if vk.CreateCommandPool(device, &compute_pool_info, allocation_callbacks, &compute_command_pool) != .SUCCESS {
            log.fatal("Failed to create compute command pool")
        }
        
        transfer_pool_info := vk.CommandPoolCreateInfo {
            sType = .COMMAND_POOL_CREATE_INFO,
            pNext = nil,
            flags = {vk.CommandPoolCreateFlag.TRANSIENT, vk.CommandPoolCreateFlag.RESET_COMMAND_BUFFER},
            queueFamilyIndex = transfer_queue_family
        }
        if vk.CreateCommandPool(device, &transfer_pool_info, allocation_callbacks, &transfer_command_pool) != .SUCCESS {
            log.fatal("Failed to create transfer command pool")
        }
    }

    // Create command buffers
    gfx_command_buffers: [dynamic]vk.CommandBuffer
    resize(&gfx_command_buffers, int(frames_in_flight))
    compute_command_buffers: [dynamic]vk.CommandBuffer
    resize(&compute_command_buffers, int(frames_in_flight))
    transfer_command_buffers: [dynamic]vk.CommandBuffer
    resize(&transfer_command_buffers, int(frames_in_flight))
    {
        gfx_info := vk.CommandBufferAllocateInfo {
            sType = .COMMAND_BUFFER_ALLOCATE_INFO,
            pNext = nil,
            commandPool = gfx_command_pool,
            level = .PRIMARY,
            commandBufferCount = frames_in_flight
        }
        if vk.AllocateCommandBuffers(device, &gfx_info, raw_data(gfx_command_buffers)) != .SUCCESS {
            log.fatal("Failed to create gfx command buffers")
        }
        compute_info := vk.CommandBufferAllocateInfo {
            sType = .COMMAND_BUFFER_ALLOCATE_INFO,
            pNext = nil,
            commandPool = compute_command_pool,
            level = .PRIMARY,
            commandBufferCount = frames_in_flight
        }
        if vk.AllocateCommandBuffers(device, &compute_info, raw_data(compute_command_buffers)) != .SUCCESS {
            log.fatal("Failed to create compute command buffers")
        }
        transfer_info := vk.CommandBufferAllocateInfo {
            sType = .COMMAND_BUFFER_ALLOCATE_INFO,
            pNext = nil,
            commandPool = transfer_command_pool,
            level = .PRIMARY,
            commandBufferCount = frames_in_flight
        }
        if vk.AllocateCommandBuffers(device, &transfer_info, raw_data(transfer_command_buffers)) != .SUCCESS {
            log.fatal("Failed to create transfer command buffers")
        }
    }

    // Create bindless descriptor set
    samplers: [TOTAL_SAMPLERS]vk.Sampler
    ds_layout: vk.DescriptorSetLayout
    dp: vk.DescriptorPool
    ds: vk.DescriptorSet
    {
        // Create immutable samplers
        {
            full_aniso_info := vk.SamplerCreateInfo {
                sType = .SAMPLER_CREATE_INFO,
                pNext = nil,
                flags = nil,
                magFilter = .LINEAR,
                minFilter = .LINEAR,
                mipmapMode = .LINEAR,
                addressModeU = .REPEAT,
                addressModeV = .REPEAT,
                addressModeW = .REPEAT,
                mipLodBias = 0.0,
                anisotropyEnable = true,
                maxAnisotropy = 16.0
            }
            vk.CreateSampler(device, &full_aniso_info, allocation_callbacks, &samplers[0])
            point_sampler_info := vk.SamplerCreateInfo {
                sType = .SAMPLER_CREATE_INFO,
                pNext = nil,
                flags = nil,
                magFilter = .NEAREST,
                minFilter = .NEAREST,
                mipmapMode = .NEAREST,
                addressModeU = .REPEAT,
                addressModeV = .REPEAT,
                addressModeW = .REPEAT,
                mipLodBias = 0.0,
                anisotropyEnable = false
            }
            vk.CreateSampler(device, &point_sampler_info, allocation_callbacks, &samplers[1])
        }

        image_binding := vk.DescriptorSetLayoutBinding {
            binding = IMAGES_DESCRIPTOR_BINDING,
            descriptorType = .SAMPLED_IMAGE,
            descriptorCount = MAXIMUM_BINDLESS_IMAGES,
            stageFlags = {.FRAGMENT},
            pImmutableSamplers = nil
        }
        sampler_binding := vk.DescriptorSetLayoutBinding {
            binding = SAMPLERS_DESCRIPTOR_BINDING,
            descriptorType = .SAMPLER,
            descriptorCount = TOTAL_SAMPLERS,
            stageFlags = {.FRAGMENT},
            pImmutableSamplers = raw_data(samplers[:])
        }
        bindings : []vk.DescriptorSetLayoutBinding = {image_binding, sampler_binding}

        layout_info := vk.DescriptorSetLayoutCreateInfo {
            sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext = nil,
            flags = {.UPDATE_AFTER_BIND_POOL},
            bindingCount = 2,
            pBindings = raw_data(bindings[:])
        }
        if vk.CreateDescriptorSetLayout(device, &layout_info, allocation_callbacks, &ds_layout) != .SUCCESS {
            log.fatal("Failed to create static descriptor set layout.")
        }

        sizes : []vk.DescriptorPoolSize = {
            vk.DescriptorPoolSize {
                type = .SAMPLED_IMAGE,
                descriptorCount = MAXIMUM_BINDLESS_IMAGES
            },
            vk.DescriptorPoolSize {
                type = .SAMPLER,
                descriptorCount = TOTAL_SAMPLERS
            }
        }

        pool_info := vk.DescriptorPoolCreateInfo {
            sType = .DESCRIPTOR_POOL_CREATE_INFO,
            pNext = nil,
            flags = {.UPDATE_AFTER_BIND},
            maxSets = 1,
            poolSizeCount = u32(len(sizes)),
            pPoolSizes = raw_data(sizes[:])
        }
        if vk.CreateDescriptorPool(device, &pool_info, allocation_callbacks, &dp) != .SUCCESS {
            log.fatal("Failed to create descriptor pool.")
        }

        set_info := vk.DescriptorSetAllocateInfo {
            sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
            pNext = nil,
            descriptorPool = dp,
            descriptorSetCount = 1,
            pSetLayouts = &ds_layout
        }
        if vk.AllocateDescriptorSets(device, &set_info, &ds) != .SUCCESS {
            log.fatal("Failed to allocate descriptor sets.")
        }
    }

    // Create global pipeline layout
    p_layout: vk.PipelineLayout
    {
        pc_range := vk.PushConstantRange {
            stageFlags = {.VERTEX,.FRAGMENT},
            offset = 0,
            size = PUSH_CONSTANTS_SIZE
        }
        layout_info := vk.PipelineLayoutCreateInfo {
            sType = .PIPELINE_LAYOUT_CREATE_INFO,
            pNext = nil,
            flags = nil,
            setLayoutCount = 1,
            pSetLayouts = &ds_layout,
            pushConstantRangeCount = 1,
            pPushConstantRanges = &pc_range
        }
        if vk.CreatePipelineLayout(device, &layout_info, allocation_callbacks, &p_layout) != .SUCCESS {
            log.fatal("Failed to create graphics pipeline layout.")
        }
    }

    gd := Graphics_Device {
        instance = inst,
        physical_device = phys_device,
        device = device,
        frames_in_flight = frames_in_flight,
        alloc_callbacks = allocation_callbacks,
        allocator = allocator,
        gfx_queue_family = gfx_queue_family,
        compute_queue_family = compute_queue_family,
        transfer_queue_family = transfer_queue_family,
        gfx_queue = gfx_queue,
        compute_queue = compute_queue,
        transfer_queue = transfer_queue,
        gfx_command_pool = gfx_command_pool,
        compute_command_pool = compute_command_pool,
        transfer_command_pool = transfer_command_pool,
        gfx_command_buffers = gfx_command_buffers,
        compute_command_buffers = compute_command_buffers,
        transfer_command_buffers = transfer_command_buffers,
        immutable_samplers = samplers,
        descriptor_set_layout = ds_layout,
        descriptor_pool = dp,
        descriptor_set = ds,
        pipeline_layout = p_layout
    }

    // Init Handle_Maps
    {
        hm.init(&gd.buffers)
        hm.init(&gd.images)
        hm.init(&gd.semaphores)
        hm.init(&gd.pipelines)
    }

    return gd
}

init_sdl2_window :: proc(gd: ^Graphics_Device, window: ^sdl2.Window) -> bool {
    if !sdl2.Vulkan_CreateSurface(window, gd.instance, &gd.surface) do return false

    width, height : i32 = 0, 0
    sdl2.Vulkan_GetDrawableSize(window, &width, &height)

    // @TODO: Allow more configurability of swapchain options
    // particularly pertaining to presentation mode and image format
    image_format := vk.Format.B8G8R8A8_SRGB
    create_info := vk.SwapchainCreateInfoKHR {
        sType = .SWAPCHAIN_CREATE_INFO_KHR,
        pNext = nil,
        flags = nil,
        surface = gd.surface,
        minImageCount = gd.frames_in_flight,
        imageFormat = image_format,
        imageColorSpace = .SRGB_NONLINEAR,
        imageExtent = vk.Extent2D {
            width = u32(width),
            height = u32(height)
        },
        imageArrayLayers = 1,
        imageUsage = {.COLOR_ATTACHMENT},
        imageSharingMode = .EXCLUSIVE,
        queueFamilyIndexCount = 1,
        pQueueFamilyIndices = &gd.gfx_queue_family,
        preTransform = {.IDENTITY},
        compositeAlpha = {.OPAQUE},
        presentMode = .FIFO,
        clipped = true,
        oldSwapchain = 0
    }
    if vk.CreateSwapchainKHR(gd.device, &create_info, gd.alloc_callbacks, &gd.swapchain) != .SUCCESS {
        return false
    }

    // Get swapchain images
    image_count : u32 = 0
    swapchain_images: [dynamic]vk.Image
    {
        vk.GetSwapchainImagesKHR(gd.device, gd.swapchain, &image_count, nil)
        resize(&swapchain_images, image_count)
        vk.GetSwapchainImagesKHR(gd.device, gd.swapchain, &image_count, raw_data(swapchain_images))
    }
    
    swapchain_image_views: [dynamic]vk.ImageView
    resize(&swapchain_image_views, image_count)
    // Create image views for the swapchain images for rendering
    {
        for vkimage, i in swapchain_images {
            info := vk.ImageViewCreateInfo {
                sType = .IMAGE_VIEW_CREATE_INFO,
                pNext = nil,
                flags = nil,
                image = vkimage,
                viewType = .D2,
                format = image_format,
                components = IDENTITY_COMPONENT_SWIZZLE,
                subresourceRange = {
                    aspectMask = {.COLOR},
                    baseMipLevel = 0,
                    levelCount = 1,
                    baseArrayLayer = 0,
                    layerCount = 1
                }
            }
            vk.CreateImageView(gd.device, &info, gd.alloc_callbacks, &swapchain_image_views[i])
        }
    }

    {
        gd := gd
        resize(&gd.swapchain_images, image_count)
        resize(&gd.acquire_semaphores, image_count)
        resize(&gd.present_semaphores, image_count)
        for i : u32 = 0; i < image_count; i += 1 {
            im := Image {
                image = swapchain_images[i],
                image_view = swapchain_image_views[i]
            }
            gd.swapchain_images[i] = Image_Handle(hm.insert(&gd.images, im))

            info := Semaphore_Info {
                type = .BINARY
            }
            gd.acquire_semaphores[i] = create_semaphore(gd, &info)
            gd.present_semaphores[i] = create_semaphore(gd, &info)
        }
    }

    return true
}

in_flight_idx :: proc(gd: ^Graphics_Device) -> u64 {
    return gd.frame_count % u64(gd.frames_in_flight)
}

Buffer_Info :: struct {
    size: vk.DeviceSize,
    usage: vk.BufferUsageFlags,
    queue_family: Queue_Family,
    required_flags: vk.MemoryPropertyFlags
}

create_buffer :: proc(gd: ^Graphics_Device, buf_info: ^Buffer_Info) -> Buffer_Handle {
    queue_family_index: u32
    switch buf_info.queue_family {
        case .Graphics: queue_family_index = gd.gfx_queue_family
        case .Compute: queue_family_index = gd.compute_queue_family
        case .Transfer: queue_family_index = gd.transfer_queue_family
    }

    info := vk.BufferCreateInfo {
        sType = .BUFFER_CREATE_INFO,
        pNext = nil,
        flags = nil,
        size = buf_info.size,
        usage = buf_info.usage,
        sharingMode = .EXCLUSIVE,
        queueFamilyIndexCount = 1,
        pQueueFamilyIndices = &queue_family_index
    }
    alloc_info := vma.Allocation_Create_Info {
        flags = nil,
        usage = .Auto,
        required_flags = buf_info.required_flags,
        preferred_flags = nil,
        priority = 1.0
    }
    b: Buffer
    if vma.create_buffer(gd.allocator, &info, &alloc_info, &b.buffer, &b.allocation, &b.alloc_info) != .SUCCESS {
        log.fatal("Failed to create buffer.")
    }
    return Buffer_Handle(hm.insert(&gd.buffers, b))
}

get_buffer :: proc(gd: ^Graphics_Device, handle: Buffer_Handle) -> (^Buffer, bool) {
    return hm.get(&gd.buffers, hm.Handle(handle))
}

delete_buffer :: proc(gd: ^Graphics_Device, handle: Buffer_Handle) -> bool {
    buffer := hm.get(&gd.buffers, hm.Handle(handle)) or_return

    buffer_delete := Buffer_Delete {
        death_frame = gd.frame_count + u64(gd.frames_in_flight),
        buffer = buffer.buffer,
        allocation = buffer.allocation
    }
    queue.append(&gd.buffer_deletes, buffer_delete)
    hm.remove(&gd.buffers, hm.Handle(handle))

    return true
}

tick_deletion_queues :: proc(gd: ^Graphics_Device) {
    // Process buffer queue
    for queue.len(gd.buffer_deletes) > 0 && queue.peek_front(&gd.buffer_deletes).death_frame == gd.frame_count {
        buffer := queue.pop_front(&gd.buffer_deletes)
        log.debugf("Destroying buffer %s...", buffer.buffer)
        vma.destroy_buffer(gd.allocator, buffer.buffer, buffer.allocation)
    }

    // @TODO: Process image queue
}

acquire_swapchain_image :: proc(gd: ^Graphics_Device, out_image_idx: ^u32) -> bool {
    idx := in_flight_idx(gd)
    sem := get_semaphore(gd, gd.acquire_semaphores[idx]) or_return
    
    if vk.AcquireNextImageKHR(gd.device, gd.swapchain, max(u64), sem^, 0, out_image_idx) != .SUCCESS {
        log.fatal("Failed to acquire swapchain image")
        return false
    }
    return true
}

present_swapchain_image :: proc(gd: ^Graphics_Device, image_idx: u32) -> bool {
    image_idx := image_idx
    idx := in_flight_idx(gd)
    sem := get_semaphore(gd, gd.present_semaphores[idx]) or_return
    info := vk.PresentInfoKHR {
        sType = .PRESENT_INFO_KHR,
        pNext = nil,
        waitSemaphoreCount = 1,
        pWaitSemaphores = sem,
        swapchainCount = 1,
        pSwapchains = &gd.swapchain,
        pImageIndices = &image_idx,
        pResults = nil
    }
    if vk.QueuePresentKHR(gd.gfx_queue, &info) != .SUCCESS {
        log.fatal("Failed to present swapchain image.")
        return false
    }
    return true
}


CommandBuffer_Index :: distinct u32

begin_gfx_command_buffer :: proc(gd: ^Graphics_Device, cpu_wait: ^Semaphore_Op) -> CommandBuffer_Index {
    cb_idx := gd.next_gfx_command_buffer
    gd.next_gfx_command_buffer = (gd.next_gfx_command_buffer + 1) % gd.frames_in_flight

    if cpu_wait.value > 0 {
        sem, ok := hm.get(&gd.semaphores, hm.Handle(cpu_wait.semaphore))
        if !ok do log.fatal("Couldn't find semaphore for CPU-sync")

        info := vk.SemaphoreWaitInfo {
            sType = .SEMAPHORE_WAIT_INFO,
            pNext = nil,
            flags = nil,
            semaphoreCount = 1,
            pSemaphores = sem,
            pValues = &cpu_wait.value
        }
        if vk.WaitSemaphores(gd.device, &info, max(u64)) != .SUCCESS {
            log.fatal("Failed to wait for timeline semaphore CPU-side man what")
        }
    }

    cb := gd.gfx_command_buffers[cb_idx]
    info := vk.CommandBufferBeginInfo {
        sType = .COMMAND_BUFFER_BEGIN_INFO,
        pNext = nil,
        flags = {.ONE_TIME_SUBMIT},
        pInheritanceInfo = nil
    }
    if vk.BeginCommandBuffer(cb, &info) != .SUCCESS {
        log.fatal("Unable to begin gfx command buffer.")
    }

    return CommandBuffer_Index(cb_idx)
}

submit_gfx_command_buffer :: proc(gd: ^Graphics_Device, cb_idx: CommandBuffer_Index, sync: ^Sync_Info) {
    cb := gd.gfx_command_buffers[cb_idx]
    if vk.EndCommandBuffer(cb) != .SUCCESS {
        log.fatal("Unable to end gfx command buffer")
    }

    cb_info := vk.CommandBufferSubmitInfo{
        sType = .COMMAND_BUFFER_SUBMIT_INFO_KHR,
        pNext = nil,
        commandBuffer = cb,
        deviceMask = 0
    }

    build_submit_infos :: proc(
        gd: ^Graphics_Device,
        submit_infos: ^[dynamic]vk.SemaphoreSubmitInfoKHR,
        semaphore_ops: ^[dynamic]Semaphore_Op
    ) -> bool {
        count := len(semaphore_ops)
        resize(submit_infos, count)
        for i := 0; i < count; i += 1 {
            sem := hm.get(&gd.semaphores, hm.Handle(semaphore_ops[i].semaphore)) or_return
            submit_infos[i] = vk.SemaphoreSubmitInfo{
                sType = .SEMAPHORE_SUBMIT_INFO_KHR,
                pNext = nil,
                semaphore = sem^,
                value = semaphore_ops[i].value,
                stageMask = {.ALL_COMMANDS},    // @TODO: This is a bit heavy-handed
                deviceIndex = 0
            }
        }
        return true
    }

    // Make semaphore submit infos
    wait_submit_infos: [dynamic]vk.SemaphoreSubmitInfoKHR
    signal_submit_infos: [dynamic]vk.SemaphoreSubmitInfoKHR
    defer delete(wait_submit_infos)
    defer delete(signal_submit_infos)
    build_submit_infos(gd, &wait_submit_infos, &sync.wait_ops)
    build_submit_infos(gd, &signal_submit_infos, &sync.signal_ops)

    info := vk.SubmitInfo2{
        sType = .SUBMIT_INFO_2_KHR,
        pNext = nil,
        flags = nil,
        waitSemaphoreInfoCount = u32(len(wait_submit_infos)),
        pWaitSemaphoreInfos = raw_data(wait_submit_infos),
        signalSemaphoreInfoCount = u32(len(signal_submit_infos)),
        pSignalSemaphoreInfos = raw_data(signal_submit_infos),
        commandBufferInfoCount = 1,
        pCommandBufferInfos = &cb_info
    }
    if vk.QueueSubmit2KHR(gd.gfx_queue, 1, &info, 0) != .SUCCESS {
        log.fatal("Unable to submit gfx command buffer")
    }
}













Framebuffer :: struct {
    color_images: [8]Image_Handle,
    depth_image: Image_Handle,
    resolution: uint2,
    clear_color: float4,
}

cmd_begin_render_pass :: proc(gd: ^Graphics_Device, cb_idx: CommandBuffer_Index, framebuffer: ^Framebuffer) {
    cb := gd.gfx_command_buffers[cb_idx]

    iv, ok := hm.get(&gd.images, hm.Handle(framebuffer.color_images[0]))
    color_attachment := vk.RenderingAttachmentInfo{
        sType = .RENDERING_ATTACHMENT_INFO_KHR,
        pNext = nil,
        imageView = iv.image_view,
        imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
        //loadOp = .DONT_CARE,
        loadOp = .CLEAR,
        storeOp = .STORE,
        clearValue = vk.ClearValue {
            color = vk.ClearColorValue {
                float32 = framebuffer.clear_color
            }
        }
    }

    info := vk.RenderingInfo{
        sType = .RENDERING_INFO_KHR,
        pNext = nil,
        flags = nil,
        renderArea = vk.Rect2D {
            extent = vk.Extent2D {
                width = framebuffer.resolution.x,
                height = framebuffer.resolution.y
            }
        },
        layerCount = 1,
        viewMask = 0,
        colorAttachmentCount = 1,
        pColorAttachments = &color_attachment,
        pDepthAttachment = nil,
        pStencilAttachment = nil
    }
    vk.CmdBeginRenderingKHR(cb, &info)
}

cmd_bind_pipeline :: proc(gd: ^Graphics_Device, cb_idx: CommandBuffer_Index, bind_point: vk.PipelineBindPoint, handle: Pipeline_Handle) -> bool {
    cb := gd.gfx_command_buffers[cb_idx]
    pipeline := hm.get(&gd.pipelines, hm.Handle(handle)) or_return
    vk.CmdBindPipeline(cb, bind_point, pipeline^)
    return true
}

cmd_draw :: proc(gd: ^Graphics_Device, cb_idx: CommandBuffer_Index, vtx_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
    cb := gd.gfx_command_buffers[cb_idx]
    vk.CmdDraw(cb, vtx_count, instance_count, first_vertex, first_instance)
}

// cmd_draw_indexed_indirect :: proc(gd: ^Graphics_Device, cb_idx: CommandBuffer_Index) {
//     cb := gd.gfx_command_buffers[cb_idx]
//     vk.CmdDrawIndexedIndirect(cb, )
// }

cmd_end_render_pass :: proc(gd: ^Graphics_Device, cb_idx: CommandBuffer_Index) {
    cb := gd.gfx_command_buffers[cb_idx]
    vk.CmdEndRenderingKHR(cb)
}

create_semaphore :: proc(gd: ^Graphics_Device, info: ^Semaphore_Info) -> Semaphore_Handle {
    t_info := vk.SemaphoreTypeCreateInfo {
        sType = .SEMAPHORE_TYPE_CREATE_INFO,
        pNext = nil,
        semaphoreType = info.type,
        initialValue = info.init_value
    }
    s_info := vk.SemaphoreCreateInfo {
        sType = .SEMAPHORE_CREATE_INFO,
        pNext = &t_info,
        flags = nil
    }
    s: vk.Semaphore
    if vk.CreateSemaphore(gd.device, &s_info, gd.alloc_callbacks, &s) != .SUCCESS {
        log.error("Failed to create semaphore.")
    }
    return Semaphore_Handle(hm.insert(&gd.semaphores, s))
}

check_timeline_semaphore :: proc(gd: ^Graphics_Device, handle: Semaphore_Handle) -> (val: u64, ok: bool) {
    sem := hm.get(&gd.semaphores, hm.Handle(handle)) or_return
    v: u64
    if vk.GetSemaphoreCounterValue(gd.device, sem^, &v) != .SUCCESS {
        log.fatal("Failed to read value from timeline semaphore")
        return 0, false
    }
    return v, true
}

get_semaphore :: proc(gd: ^Graphics_Device, handle: Semaphore_Handle) -> (^vk.Semaphore, bool) {
    return hm.get(&gd.semaphores, hm.Handle(handle))
}

destroy_semaphore :: proc(gd: ^Graphics_Device, handle: Semaphore_Handle) -> bool {
    semaphore := hm.get(&gd.semaphores, hm.Handle(handle)) or_return
    vk.DestroySemaphore(gd.device, semaphore^, gd.alloc_callbacks)
    
    return true
}

// Defines a memory dependency on exactly
// one subresource range of a vk.Image (VkImage)
Image_Barrier :: struct {
    src_stage_mask: vk.PipelineStageFlags2,
    src_access_mask: vk.AccessFlags2,
    dst_stage_mask: vk.PipelineStageFlags2,
    dst_access_mask: vk.AccessFlags2,
    old_layout: vk.ImageLayout,
    new_layout: vk.ImageLayout,
    src_queue_family: u32,
    dst_queue_family: u32,
    image_handle: Image_Handle,
    subresource_range: vk.ImageSubresourceRange
}

// Inserts an arbitrary number of memory barriers
// into the command buffer at this point
cmd_pipeline_barrier :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    image_barriers: []Image_Barrier
) {
    cb := gd.gfx_command_buffers[cb_idx]

    im_barriers: [dynamic]vk.ImageMemoryBarrier2
    defer delete(im_barriers)
    reserve(&im_barriers, len(image_barriers))
    for barrier in image_barriers {
        using barrier

        im := hm.get(&gd.images, hm.Handle(image_handle)) or_continue   // @TODO: Should this really be or_continue?
        append(
            &im_barriers,
            vk.ImageMemoryBarrier2 {
                sType = .IMAGE_MEMORY_BARRIER_2,
                pNext = nil,
                srcStageMask = src_stage_mask,
                srcAccessMask = src_access_mask,
                dstStageMask = dst_stage_mask,
                dstAccessMask = dst_access_mask,
                oldLayout = old_layout,
                newLayout = new_layout,
                srcQueueFamilyIndex = src_queue_family,
                dstQueueFamilyIndex = dst_queue_family,
                image = im.image,
                subresourceRange = subresource_range
            }
        )
    }

    info := vk.DependencyInfo {
        sType = .DEPENDENCY_INFO,
        pNext = nil,
        dependencyFlags = nil,
        memoryBarrierCount = 0,
        pMemoryBarriers = nil,
        bufferMemoryBarrierCount = 0,
        pBufferMemoryBarriers = nil,
        imageMemoryBarrierCount = u32(len(im_barriers)),
        pImageMemoryBarriers = raw_data(im_barriers)
    }
    vk.CmdPipelineBarrier2KHR(cb, &info)
}






// Graphics pipeline section
// Using a unified pipeline layout

Input_Assembly_State :: struct {
    // flags: PipelineInputAssemblyStateCreateFlags,
    topology: vk.PrimitiveTopology,
    primitive_restart_enabled: bool
}

Tessellation_State :: struct {
    patch_control_points: u32
}

Rasterization_State :: struct {
    do_depth_clamp: bool,
    do_rasterizer_discard: bool,
    polygon_mode: vk.PolygonMode,
    cull_mode: vk.CullModeFlags,
    front_face: vk.FrontFace,
    do_depth_bias: bool,
    depth_bias_constant_factor: f32,
	depth_bias_clamp:          f32,
	depth_bias_slope_factor:    f32,
	line_width:               f32
}

default_rasterization_state :: proc() -> Rasterization_State {
    return Rasterization_State {
        do_depth_clamp = false,
        do_rasterizer_discard = false,
        polygon_mode = .FILL,
        cull_mode = {.BACK},
        front_face = .COUNTER_CLOCKWISE,
        do_depth_bias = false,
        depth_bias_constant_factor = 0.0,
        depth_bias_clamp = 0.0,
        depth_bias_slope_factor = 0.0,
        line_width = 1.0
    }
}

Multisample_State :: struct {
    sample_count: vk.SampleCountFlags,
    do_sample_shading: bool,
    min_sample_shading: f32,
    sample_mask: vk.SampleMask,
    do_alpha_to_coverage: bool,
    do_alpha_to_one: bool
}

DepthStencil_State :: struct {
    flags: vk.PipelineDepthStencilStateCreateFlags,
    do_depth_test: bool,
    do_depth_write: bool,
    depth_compare_op: vk.CompareOp,
    do_depth_bounds_test: bool,
    do_stencil_test: bool,
    front: vk.StencilOpState,
    back: vk.StencilOpState,
    min_depth_bounds: f32,
    max_depth_bounds: f32,
}

ColorBlend_Attachment :: struct {
    do_blend: bool,
    src_color_blend_factor: vk.BlendFactor,
    dst_color_blend_factor: vk.BlendFactor,
    color_blend_op: vk.BlendOp,
    src_alpha_blend_factor: vk.BlendFactor,
    dst_alpha_blend_factor: vk.BlendFactor,
    alpha_blend_op: vk.BlendOp,
    color_write_mask: vk.ColorComponentFlags
}

ColorBlend_State :: struct {
    flags: vk.PipelineColorBlendStateCreateFlags,
    do_logic_op: bool,
    logic_op: vk.LogicOp,
    blend_constants: float4,
    attachment: ColorBlend_Attachment
    //attachments: [dynamic]ColorBlend_Attachment
}

default_colorblend_state :: proc() -> ColorBlend_State {
    // attachments: [dynamic]ColorBlend_Attachment
    // append(&attachments, ColorBlend_Attachment {
    //     do_blend = true,
    //     src_color_blend_factor = .SRC_ALPHA,
    //     dst_color_blend_factor = .ONE_MINUS_SRC_ALPHA,
    //     color_blend_op = .ADD,
    //     src_alpha_blend_factor = .SRC_ALPHA,
    //     dst_alpha_blend_factor = .ONE_MINUS_SRC_ALPHA,
    //     alpha_blend_op = .ADD,
    //     color_write_mask = {.R,.G,.B,.A}
    // })

    return ColorBlend_State {
        flags = nil,
        do_logic_op = false,
        logic_op = nil,
        blend_constants = {1.0, 1.0, 1.0, 1.0},
        attachment = ColorBlend_Attachment {
            do_blend = true,
            src_color_blend_factor = .SRC_ALPHA,
            dst_color_blend_factor = .ONE_MINUS_SRC_ALPHA,
            color_blend_op = .ADD,
            src_alpha_blend_factor = .SRC_ALPHA,
            dst_alpha_blend_factor = .ONE_MINUS_SRC_ALPHA,
            alpha_blend_op = .ADD,
            color_write_mask = {.R,.G,.B,.A}
        }
        //attachments = attachments
    }
}

PipelineRenderpass_Info :: struct {
    color_attachment_formats: []vk.Format,
    depth_attachment_format: vk.Format
}

Pipeline_Handle :: distinct hm.Handle
Graphics_Pipeline_Info :: struct {
    vertex_shader_bytecode: []u32,
    fragment_shader_bytecode: []u32,
    input_assembly_state: Input_Assembly_State,
    tessellation_state: Tessellation_State,
    rasterization_state: Rasterization_State,
    multisample_state: Multisample_State,
    depthstencil_state: DepthStencil_State,
    colorblend_state: ColorBlend_State,
    renderpass_state: PipelineRenderpass_Info
}

create_graphics_pipelines :: proc(gd: ^Graphics_Device, infos: []Graphics_Pipeline_Info) -> [dynamic]Pipeline_Handle {
    pipeline_count := len(infos)

    // Output dynamic array of pipeline handles
    handles: [dynamic]Pipeline_Handle
    resize(&handles, pipeline_count)

    // One dynamic array for each thing in Graphics_Pipeline_Info
    create_infos: [dynamic]vk.GraphicsPipelineCreateInfo
    pipelines: [dynamic]vk.Pipeline
    shader_module_infos: [dynamic]vk.ShaderModuleCreateInfo
    shader_infos: [dynamic]vk.PipelineShaderStageCreateInfo
    input_assembly_states: [dynamic]vk.PipelineInputAssemblyStateCreateInfo
    tessellation_states: [dynamic]vk.PipelineTessellationStateCreateInfo
    rasterization_states: [dynamic]vk.PipelineRasterizationStateCreateInfo
    multisample_states: [dynamic]vk.PipelineMultisampleStateCreateInfo
    sample_masks: [dynamic]vk.SampleMask
    depthstencil_states: [dynamic]vk.PipelineDepthStencilStateCreateInfo
    colorblend_attachments: [dynamic]vk.PipelineColorBlendAttachmentState
    colorblend_states: [dynamic]vk.PipelineColorBlendStateCreateInfo
    renderpass_states: [dynamic]vk.PipelineRenderingCreateInfo
    defer delete(renderpass_states)
    defer delete(shader_module_infos)
    defer delete(shader_infos)
    defer delete(colorblend_attachments)
    defer delete(colorblend_states)
    defer delete(depthstencil_states)
    defer delete(sample_masks)
    defer delete(multisample_states)
    defer delete(tessellation_states)
    defer delete(rasterization_states)
    defer delete(input_assembly_states)
    defer delete(create_infos)
    defer delete(pipelines)
    resize(&create_infos, pipeline_count)
    resize(&renderpass_states, pipeline_count)
    resize(&pipelines, pipeline_count)
    resize(&shader_module_infos, 2 * pipeline_count)
    resize(&shader_infos, 2 * pipeline_count)
    resize(&input_assembly_states, pipeline_count)
    resize(&tessellation_states, pipeline_count)
    resize(&rasterization_states, pipeline_count)
    resize(&multisample_states, pipeline_count)
    resize(&sample_masks, pipeline_count)
    resize(&depthstencil_states, pipeline_count)
    resize(&colorblend_attachments, pipeline_count)
    resize(&colorblend_states, pipeline_count)
    
    dynamic_states : [2]vk.DynamicState = {.VIEWPORT,.SCISSOR}
    
    // Constant pipeline create infos
    vertex_input_info := vk.PipelineVertexInputStateCreateInfo {
        sType = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        pNext = nil,
        flags = nil,
        vertexBindingDescriptionCount = 0,
        pVertexBindingDescriptions = nil,
        vertexAttributeDescriptionCount = 0,
        pVertexAttributeDescriptions = nil
    }

    viewport_info := vk.PipelineViewportStateCreateInfo {
        sType = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        pNext = nil,
        flags = nil,
        viewportCount = 1,
        pViewports = nil,
        scissorCount = 1,
        pScissors = nil
    } 

    dynamic_state_info := vk.PipelineDynamicStateCreateInfo {
        sType = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        pNext = nil,
        flags = nil,
        dynamicStateCount = len(dynamic_states),
        pDynamicStates = raw_data(dynamic_states[:])
    }

    // Make create infos
    for info, i in infos {
        using info

        // Shader state
        shader_module_infos[2 * i] = vk.ShaderModuleCreateInfo {
            sType = .SHADER_MODULE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            codeSize = size_of(u32) * len(vertex_shader_bytecode),
            pCode = raw_data(vertex_shader_bytecode)
        }
        shader_module_infos[2 * i + 1] = vk.ShaderModuleCreateInfo {
            sType = .SHADER_MODULE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            codeSize = size_of(u32) * len(fragment_shader_bytecode),
            pCode = raw_data(fragment_shader_bytecode)
        }
        shader_infos[2 * i] = vk.PipelineShaderStageCreateInfo {
            sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
            pNext = &shader_module_infos[2 * i],
            flags = nil,
            stage = {.VERTEX},
            module = 0,
            pName = "main",
            pSpecializationInfo = nil
        }
        shader_infos[2 * i + 1] = vk.PipelineShaderStageCreateInfo {
            sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
            pNext = &shader_module_infos[2 * i + 1],
            flags = nil,
            stage = {.FRAGMENT},
            module = 0,
            pName = "main",
            pSpecializationInfo = nil
        }

        // Input assembly state
        input_assembly_states[i] = vk.PipelineInputAssemblyStateCreateInfo {
            sType = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            topology = input_assembly_state.topology,
            primitiveRestartEnable = b32(input_assembly_state.primitive_restart_enabled)
        }

        // Tessellation state
        tessellation_states[i] = vk.PipelineTessellationStateCreateInfo {
            sType = .PIPELINE_TESSELLATION_STATE_CREATE_INFO,
            pNext = nil,
            patchControlPoints = tessellation_state.patch_control_points
        }

        // Rasterization state
        rasterization_states[i] = vk.PipelineRasterizationStateCreateInfo {
            sType = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            depthClampEnable = b32(rasterization_state.do_depth_clamp),
            rasterizerDiscardEnable = b32(rasterization_state.do_rasterizer_discard),
            polygonMode = rasterization_state.polygon_mode,
            cullMode = rasterization_state.cull_mode,
            frontFace = rasterization_state.front_face,
            depthBiasEnable = b32(rasterization_state.do_depth_bias),
            depthBiasConstantFactor = rasterization_state.depth_bias_constant_factor,
            depthBiasClamp = rasterization_state.depth_bias_clamp,
            depthBiasSlopeFactor = rasterization_state.depth_bias_slope_factor,
            lineWidth = rasterization_state.line_width
        }

        // Multisample state
        sample_masks[i] = multisample_state.sample_mask
        multisample_states[i] = vk.PipelineMultisampleStateCreateInfo {
            sType = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            rasterizationSamples = multisample_state.sample_count,
            sampleShadingEnable = b32(multisample_state.do_sample_shading),
            minSampleShading = multisample_state.min_sample_shading,
            pSampleMask = &sample_masks[i]
        }

        // Depth-stencil state
        depthstencil_states[i] = vk.PipelineDepthStencilStateCreateInfo {
            sType = .PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            pNext = nil,
            flags = depthstencil_state.flags,
            depthTestEnable = b32(depthstencil_state.do_depth_test),
            depthWriteEnable = b32(depthstencil_state.do_depth_write),
            depthCompareOp = depthstencil_state.depth_compare_op,
            depthBoundsTestEnable = b32(depthstencil_state.do_depth_bounds_test),
            stencilTestEnable = b32(depthstencil_state.do_stencil_test),
            front = depthstencil_state.front,
            back = depthstencil_state.back,
            minDepthBounds = depthstencil_state.min_depth_bounds,
            maxDepthBounds = depthstencil_state.max_depth_bounds
        }

        // Color blend state
        colorblend_attachments[i] = vk.PipelineColorBlendAttachmentState {
            blendEnable = b32(colorblend_state.attachment.do_blend),
            srcColorBlendFactor = colorblend_state.attachment.src_color_blend_factor,
            dstColorBlendFactor = colorblend_state.attachment.dst_color_blend_factor,
            colorBlendOp = colorblend_state.attachment.color_blend_op,
            srcAlphaBlendFactor = colorblend_state.attachment.src_alpha_blend_factor,
            dstAlphaBlendFactor = colorblend_state.attachment.dst_alpha_blend_factor,
            alphaBlendOp = colorblend_state.attachment.alpha_blend_op,
            colorWriteMask = colorblend_state.attachment.color_write_mask
        }
        colorblend_states[i] = vk.PipelineColorBlendStateCreateInfo {
            sType = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            pNext = nil,
            flags = colorblend_state.flags,
            logicOpEnable = b32(colorblend_state.do_logic_op),
            attachmentCount = 1,
            pAttachments = &colorblend_attachments[i],
            blendConstants = colorblend_state.blend_constants
        }

        // Render pass state
        renderpass_states[i] = vk.PipelineRenderingCreateInfo {
            sType = .PIPELINE_RENDERING_CREATE_INFO,
            pNext = nil,
            viewMask = 0,
            colorAttachmentCount = u32(len(renderpass_state.color_attachment_formats)),
            pColorAttachmentFormats = raw_data(renderpass_state.color_attachment_formats),
            depthAttachmentFormat = renderpass_state.depth_attachment_format,
            stencilAttachmentFormat = nil
        }

        create_infos[i] = vk.GraphicsPipelineCreateInfo {
            sType = .GRAPHICS_PIPELINE_CREATE_INFO,
            pNext = &renderpass_states[i],
            flags = nil,
            stageCount = 2,
            pStages = &shader_infos[2 * i],
            pVertexInputState = &vertex_input_info,         // Always manually pull vertices in the vertex shader
            pInputAssemblyState = &input_assembly_states[i],
            pTessellationState = &tessellation_states[i],
            pViewportState = &viewport_info,
            pRasterizationState = &rasterization_states[i],
            pMultisampleState = &multisample_states[i],
            pDepthStencilState = &depthstencil_states[i],
            pColorBlendState = &colorblend_states[i],
            pDynamicState = &dynamic_state_info,
            layout = gd.pipeline_layout,
            renderPass = 0,
            subpass = 0,
            basePipelineHandle = 0,
            basePipelineIndex = 0
        }
    }

    res := vk.CreateGraphicsPipelines(
        gd.device,
        gd.pipeline_cache,
        u32(pipeline_count),
        raw_data(create_infos),
        gd.alloc_callbacks,
        raw_data(pipelines)
    )
    if res != .SUCCESS {
        log.fatal("Failed to compile graphics pipelines")
    }

    // Put newly created pipelines in the Handle_Map
    for p, i in pipelines {
        handles[i] = Pipeline_Handle(hm.insert(&gd.pipelines, p))
    }

    return handles
}