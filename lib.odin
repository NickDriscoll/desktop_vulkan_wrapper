package desktop_vulkan_wrapper

import "core:container/queue"
import "core:fmt"
import "core:log"
import "core:math"
import "core:math/linalg/hlsl"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strings"

import "vendor:sdl2"
import vk "vendor:vulkan"

import "odin-vma/vma"
import hm "handlemap"

MAXIMUM_BINDLESS_IMAGES :: 1024 * 1024

// Sizes in bytes
PUSH_CONSTANTS_SIZE :: 128
STAGING_BUFFER_SIZE :: 16 * 1024 * 1024
AS_BUFFER_SIZE :: 16 * 1024 * 1024
TLAS_INSTANCE_BUFFER_SIZE :: 16 * 1024

PIPELINE_CACHE_FILENAME :: ".shadercache"

create_write_file :: proc(filename: string) -> (os.Handle, os.Error) {
    h: os.Handle

    err: os.Errno
    when ODIN_OS == .Windows {
        h, err = os.open(
            filename,
            os.O_WRONLY | os.O_CREATE | os.O_TRUNC
        )
    }
    when ODIN_OS == .Linux {
        h, err = os.open(
            filename,
            os.O_WRONLY | os.O_CREATE | os.O_TRUNC,
            os.S_IRUSR | os.S_IWUSR
        )
    }

    return h, err
}

size_to_alignment :: proc(size: $T, alignment: T) -> T {
    assert(alignment > 0)
    return (size + (alignment - 1)) & ~(alignment - 1)
}

// All buffers are read/written by the GPU using Buffer Device Address
// As such, we only need descriptor slots for images and samplers
Bindless_Descriptor_Bindings :: enum u32 {
    Images = 0,
    Samplers = 1,
    AccelerationStructures = 2,
}
TOTAL_AS_DESCRIPTORS :: 4

Immutable_Sampler_Index :: enum u32 {
    Aniso16 = 0,
    Point = 1,
    PostFX = 2
}
TOTAL_IMMUTABLE_SAMPLERS :: len(Immutable_Sampler_Index)

IDENTITY_COMPONENT_SWIZZLE :: vk.ComponentMapping {
    r = .R,
    g = .G,
    b = .B,
    a = .A,
}

Queue_Family :: enum {
    Graphics,
    Compute,
    Transfer
}

Semaphore_Op :: struct {
    semaphore: Semaphore_Handle,
    value: u64
}

SyncInfo :: struct {
    wait_ops: [dynamic]Semaphore_Op,
    signal_ops: [dynamic]Semaphore_Op,
}

sync_init :: proc(s: ^SyncInfo) {
    s.wait_ops = make([dynamic]Semaphore_Op)
    s.signal_ops = make([dynamic]Semaphore_Op)
}

add_wait_op :: proc(gd: ^Graphics_Device, i: ^SyncInfo, handle: Semaphore_Handle, value : u64 = 0) {
    append(&i.wait_ops, Semaphore_Op {
        semaphore = handle,
        value = value
    })
}
add_signal_op :: proc(gd: ^Graphics_Device, i: ^SyncInfo, handle: Semaphore_Handle, value : u64 = 0) {
    append(&i.signal_ops, Semaphore_Op {
        semaphore = handle,
        value = value
    })
}

delete_sync_info :: proc(s: ^SyncInfo) {
    delete(s.wait_ops)
    delete(s.signal_ops)
}

clear_sync_info :: proc(s: ^SyncInfo) {
    clear(&s.wait_ops)
    clear(&s.signal_ops)
}

SupportFlags :: bit_set[enum {
    Window,              // Will this device need to draw to window surface swapchains?
    Raytracing,          // Will this device need raytracing capabilities?
}]

// Distinct handle types for each Handle_Map in the Graphics_Device
Buffer_Handle :: distinct hm.Handle
Texture_Handle :: distinct hm.Handle
Acceleration_Structure_Handle :: distinct hm.Handle
Semaphore_Handle :: distinct hm.Handle

// Megastruct holding basically all Vulkan-specific state
Graphics_Device :: struct {
    // Basic Vulkan state that every app definitely needs
    instance: vk.Instance,
    physical_device: vk.PhysicalDevice,
    physical_device_properties: vk.PhysicalDeviceProperties2,
    device: vk.Device,
    pipeline_cache: vk.PipelineCache,
    alloc_callbacks: ^vk.AllocationCallbacks,
    allocator: vma.Allocator,

    support_flags: SupportFlags,

    frames_in_flight: u32,
    frame_count: u64,

    // Objects required to support windowing
    // Basically every app will use these, but maybe
    // these could be factored out
    surface: vk.SurfaceKHR,
    swapchain: vk.SwapchainKHR,
    swapchain_images: [dynamic]Texture_Handle,
    acquire_semaphores: [dynamic]Semaphore_Handle,
    present_semaphores: [dynamic]Semaphore_Handle,
    resize_window: bool,

    // The Vulkan queues that the device will submit on
    // May be aliases of each other if the GPU doesn't have
    // e.g. an async compute queue or dedicated transfer queue
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
    next_compute_command_buffer: u32,

    // All users of this Vulkan wrapper will access
    // buffers via their device addresses and
    // images through this global descriptor set
    // i.e. all bindless all the time, baby
    immutable_samplers: [TOTAL_IMMUTABLE_SAMPLERS]vk.Sampler,
    descriptor_set_layout: vk.DescriptorSetLayout,
    descriptor_pool: vk.DescriptorPool,
    descriptor_set: vk.DescriptorSet,

    // Host-mappable GPU buffer used for doing data
    // transfer between host and device memory
    staging_buffer: Buffer_Handle,
    staging_buffer_offest: u64,
    transfer_timeline: Semaphore_Handle,
    transfers_completed: u64,

    // Acceleration structure state
    AS_buffer: Buffer_Handle,                                               // Stores built acceleration structures
    BLAS_head: vk.DeviceSize,                                                 // Points to location of next BLAS in AS_buffer
    AS_scratch_buffer: Buffer_Handle,                                       // Scratch buffer for AS builds
    AS_scratch_size: vk.DeviceSize,                                         // Current size of AS_scratch_buffer
    BLAS_queued_build_infos: queue.Queue(AccelerationStructureBuildInfo),     // Queue of bottom-level acceleration structures to build this frame
    AS_required_scratch_size: vk.DeviceSize,                                // Current total scratch size required by queued build infos
    TLAS_instance_buffer: Buffer_Handle,
    
    // Pipeline layouts used for all pipelines
    gfx_pipeline_layout: vk.PipelineLayout,
    compute_pipeline_layout: vk.PipelineLayout,

    // Handle_Maps of all Vulkan objects
    buffers: hm.Handle_Map(Buffer),
    images: hm.Handle_Map(Image),
    acceleration_structures: hm.Handle_Map(AccelerationStructure),
    semaphores: hm.Handle_Map(vk.Semaphore),
    pipelines: hm.Handle_Map(vk.Pipeline),

    // Queue for images who's data has been uploaded
    // but which haven't been transferred to the gfx queue
    pending_images: queue.Queue(Pending_Image),

    // Deletion queues for resources
    buffer_deletes: queue.Queue(Buffer_Delete),
    image_deletes: queue.Queue(Image_Delete),
    AS_deletes: queue.Queue(AS_Delete),

}

// @TODO: What am I doing with this?
debug_utils_callback :: proc "system" (
    messageSeverity: vk.DebugUtilsMessageSeverityFlagsEXT,
    messageTypes: vk.DebugUtilsMessageTypeFlagsEXT,
    pCallbackData: ^vk.DebugUtilsMessengerCallbackDataEXT,
    pUserData: rawptr
) -> b32 {
    message := pCallbackData.pMessage
    return true
}

string_from_bytes :: proc(bytes: []u8) -> string {
    return strings.string_from_null_terminated_ptr(&bytes[0], len(bytes))
}

Init_Parameters :: struct {
    // Vulkan instance creation parameters
    app_name: cstring,
    app_version: u32,
    engine_name: cstring,
    engine_version: u32,

    allocation_callbacks: ^vk.AllocationCallbacks,
    vk_get_instance_proc_addr: rawptr,

    frames_in_flight: u32,      // Maximum number of command buffers active at once

    features: SupportFlags
}

init_vulkan :: proc(params: Init_Parameters) -> (Graphics_Device, vk.Result) {
    assert(params.frames_in_flight > 0)

    comp_bytes_to_string :: proc(bytes: []byte, s: string) -> bool {
        return string(bytes[0:len(s)]) == s
    }

    string_contained :: proc(list: []vk.ExtensionProperties, cand: string) -> bool {
        for ext in list {
            name := ext.extensionName
            if comp_bytes_to_string(name[:], cand) {
                return true
            }
        }
        return false
    }

    gd: Graphics_Device
    gd.frames_in_flight = params.frames_in_flight
    gd.alloc_callbacks = params.allocation_callbacks

    log.info("Initializing Vulkan instance and device")

    if params.vk_get_instance_proc_addr == nil {
        log.error("Init_Paramenters.vk_get_instance_proc_addr was nil!")
    }
    vk.load_proc_addresses_global(params.vk_get_instance_proc_addr)

    // Create Vulkan instance
    // @TODO: Look into vkEnumerateInstanceVersion()
    api_version_int : u32 = vk.API_VERSION_1_3
    {
        // Instead of forcing the caller to explicitly provide
        // the extensions they want to enable, I want to provide high-level
        // idioms that cover many extensions in the same logical category

        ext_count: u32 
        vk.EnumerateInstanceExtensionProperties(nil, &ext_count, nil)
        supported_extensions := make([dynamic]vk.ExtensionProperties, ext_count, context.temp_allocator)
        vk.EnumerateInstanceExtensionProperties(nil, &ext_count, raw_data(supported_extensions))

        final_extensions := make([dynamic]cstring, 0, 16, context.temp_allocator)
        if .Window in params.features {
            exts: []string
            when ODIN_OS == .Windows {
                exts = {vk.KHR_SURFACE_EXTENSION_NAME, vk.KHR_WIN32_SURFACE_EXTENSION_NAME}
            }
            when ODIN_OS == .Linux {
                exts = {vk.KHR_SURFACE_EXTENSION_NAME, vk.KHR_XLIB_SURFACE_EXTENSION_NAME}
            }

            for ext in exts {
                if string_contained(supported_extensions[:], ext) {
                    append(&final_extensions, strings.clone_to_cstring(ext, context.temp_allocator))
                } else {
                    log.errorf("Required instance extension %v not found.", ext)
                }
            }
        }

        append(&final_extensions, vk.EXT_DEBUG_UTILS_EXTENSION_NAME)
        debug_info := vk.DebugUtilsMessengerCreateInfoEXT {
            sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            pNext = nil,
            flags = nil,
            messageSeverity = {.ERROR,.WARNING},
            messageType = {.GENERAL,.VALIDATION},
            pfnUserCallback = debug_utils_callback,
            pUserData = nil
        }

        app_info := vk.ApplicationInfo {
            sType = .APPLICATION_INFO,
            pNext = nil,
            pApplicationName = params.app_name,
            applicationVersion = params.app_version,
            pEngineName = params.engine_name,
            engineVersion = params.engine_version,
            apiVersion = api_version_int
        }
        create_info := vk.InstanceCreateInfo {
            sType = .INSTANCE_CREATE_INFO,
            pNext = &debug_info,
            flags = nil,
            pApplicationInfo = &app_info,
            enabledLayerCount = 0,
            ppEnabledLayerNames = nil,
            enabledExtensionCount = u32(len(final_extensions)),
            ppEnabledExtensionNames = raw_data(final_extensions)
        }

        r := vk.CreateInstance(&create_info, params.allocation_callbacks, &gd.instance)
        if r != .SUCCESS {
            log.error("Instance creation failed.")
            return gd, r
        }
    }

    // Load instance-level procedures
    vk.load_proc_addresses_instance(gd.instance)

    // Create Vulkan device
    {
        phys_device_count : u32 = 0
        vk.EnumeratePhysicalDevices(gd.instance, &phys_device_count, nil)
        phys_devices := make([dynamic]vk.PhysicalDevice, phys_device_count, context.temp_allocator)
        vk.EnumeratePhysicalDevices(gd.instance, &phys_device_count, raw_data(phys_devices))

        preferred_device_types : []vk.PhysicalDeviceType = {.DISCRETE_GPU,.INTEGRATED_GPU}

        // Select the physical device to use
        // @NOTE: We only support using a single physical device at once
        features: vk.PhysicalDeviceFeatures2
        outer: for device_type in preferred_device_types {
            for pd in phys_devices {
                // Query this physical device's properties
                vk12_props: vk.PhysicalDeviceVulkan12Properties
                props: vk.PhysicalDeviceProperties2
    
                vk12_props.sType = .PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES
                props.sType = .PHYSICAL_DEVICE_PROPERTIES_2
    
                props.pNext = &vk12_props
                vk.GetPhysicalDeviceProperties2(pd, &props)
    
                // @TODO: Do something more sophisticated than picking the first device of a preferred type
                if props.properties.deviceType == device_type {
                    log.debugf("Considering physical device:\t%v", string_from_bytes(props.properties.deviceName[:]))

                    // Check physical device features
                    vulkan_11_features: vk.PhysicalDeviceVulkan11Features
                    vulkan_12_features: vk.PhysicalDeviceVulkan12Features
                    dynamic_rendering_features: vk.PhysicalDeviceDynamicRenderingFeatures
                    sync2_features: vk.PhysicalDeviceSynchronization2Features
                    accel_features: vk.PhysicalDeviceAccelerationStructureFeaturesKHR
                    rt_pipeline_features: vk.PhysicalDeviceRayTracingPipelineFeaturesKHR
                    ray_query_features: vk.PhysicalDeviceRayQueryFeaturesKHR
    
                    vulkan_11_features.sType = .PHYSICAL_DEVICE_VULKAN_1_1_FEATURES
                    vulkan_12_features.sType = .PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
                    dynamic_rendering_features.sType = .PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES
                    sync2_features.sType = .PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES
                    accel_features.sType = .PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR
                    rt_pipeline_features.sType = .PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR
                    ray_query_features.sType = .PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR
                    features.sType = .PHYSICAL_DEVICE_FEATURES_2
    
                    rt_pipeline_features.pNext = &ray_query_features
                    accel_features.pNext = &rt_pipeline_features
                    vulkan_11_features.pNext = &accel_features
                    vulkan_12_features.pNext = &vulkan_11_features
                    dynamic_rendering_features.pNext = &vulkan_12_features
                    sync2_features.pNext = &dynamic_rendering_features
                    features.pNext = &sync2_features
                    vk.GetPhysicalDeviceFeatures2(pd, &features)
    
                    // Check for raytracing features
                    has_right_features : b32 = true
                    if .Raytracing in params.features {
                        gd.support_flags += {.Raytracing}
                        has_right_features = accel_features.accelerationStructure &&
                                             accel_features.descriptorBindingAccelerationStructureUpdateAfterBind &&
                                             rt_pipeline_features.rayTracingPipeline &&
                                             ray_query_features.rayQuery
    
                        if !has_right_features {
                            gd.support_flags -= {.Raytracing}
                            vulkan_11_features.pNext = nil
                        }
                    } else {
                        vulkan_11_features.pNext = nil
                    }
    
                    has_right_features =
                        vulkan_11_features.variablePointers &&
                        vulkan_12_features.descriptorIndexing &&
                        vulkan_12_features.runtimeDescriptorArray &&
                        vulkan_12_features.timelineSemaphore &&
                        vulkan_12_features.bufferDeviceAddress && 
                        dynamic_rendering_features.dynamicRendering &&
                        sync2_features.synchronization2 
                    if has_right_features {
                        gd.physical_device = pd
                        gd.physical_device_properties = props
                        log.infof("Chosen GPU: %s", string_from_bytes(props.properties.deviceName[:]))
                        break outer
                    }
                }
            }
        }
        assert(gd.physical_device != nil, "Didn't find VkPhysicalDevice")

        // Query the physical device's queue family properties
        queue_family_count : u32 = 0
        vk.GetPhysicalDeviceQueueFamilyProperties2(gd.physical_device, &queue_family_count, nil)

        qfps := make([dynamic]vk.QueueFamilyProperties2, queue_family_count, context.temp_allocator)
        for &qfp in qfps {
            qfp.sType = .QUEUE_FAMILY_PROPERTIES_2
        }
        vk.GetPhysicalDeviceQueueFamilyProperties2(gd.physical_device, &queue_family_count, raw_data(qfps))

        // Determine available queue family types
        for qfp, i in qfps {
            flags := qfp.queueFamilyProperties.queueFlags
            if vk.QueueFlag.GRAPHICS in flags {
                gd.gfx_queue_family = u32(i)
            }
        }
        gd.compute_queue_family = gd.gfx_queue_family
        gd.transfer_queue_family = gd.gfx_queue_family
        for qfp, i in qfps {
            flags := qfp.queueFamilyProperties.queueFlags

            if .COMPUTE&.TRANSFER in flags {
                gd.compute_queue_family = u32(i)
                gd.transfer_queue_family = u32(i)
            } else if vk.QueueFlag.COMPUTE in flags {
                gd.compute_queue_family = u32(i)
            } else if vk.QueueFlag.TRANSFER in flags {
                gd.transfer_queue_family = u32(i)
            }
        }

        queue_priority : f32 = 1.0
        queue_count : u32 = 1
        queue_create_infos: [3]vk.DeviceQueueCreateInfo
        queue_create_infos[0] = vk.DeviceQueueCreateInfo {
            sType = .DEVICE_QUEUE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            queueFamilyIndex = gd.gfx_queue_family,
            queueCount = 1,
            pQueuePriorities = &queue_priority
        }
        if gd.compute_queue_family != gd.gfx_queue_family {
            queue_count += 1
            queue_create_infos[1] = vk.DeviceQueueCreateInfo {
                sType = .DEVICE_QUEUE_CREATE_INFO,
                pNext = nil,
                flags = nil,
                queueFamilyIndex = gd.compute_queue_family,
                queueCount = 1,
                pQueuePriorities = &queue_priority
            }
        }
        if gd.transfer_queue_family != gd.compute_queue_family {
            queue_count += 1
            queue_create_infos[2] = vk.DeviceQueueCreateInfo {
                sType = .DEVICE_QUEUE_CREATE_INFO,
                pNext = nil,
                flags = nil,
                queueFamilyIndex = gd.transfer_queue_family,
                queueCount = 1,
                pQueuePriorities = &queue_priority
            }
        }

        // Device extensions

        //Load all supported device extensions for later querying
        extension_count : u32 = 0
        vk.EnumerateDeviceExtensionProperties(gd.physical_device, nil, &extension_count, nil)
        supported_extensions := make([dynamic]vk.ExtensionProperties, extension_count, context.temp_allocator)
        vk.EnumerateDeviceExtensionProperties(gd.physical_device, nil, &extension_count, raw_data(supported_extensions))

        required_extensions: []string = {
            vk.EXT_MEMORY_BUDGET_EXTENSION_NAME
        }

        final_extensions := make([dynamic]cstring, 0, 16, context.temp_allocator)
        for ext in required_extensions {
            if !string_contained(supported_extensions[:], ext) {
                log.errorf("Device doesn't have required extension: %v", ext)
                return gd, .ERROR_EXTENSION_NOT_PRESENT
            }
            append(&final_extensions, strings.clone_to_cstring(ext, context.temp_allocator))
        }
        if .Window in params.features {
            gd.support_flags += {.Window}
            if string_contained(supported_extensions[:], vk.KHR_SWAPCHAIN_EXTENSION_NAME) {
                append(&final_extensions, vk.KHR_SWAPCHAIN_EXTENSION_NAME)
            } else {
                log.errorf("Requested window support but %v was not found", vk.KHR_SWAPCHAIN_EXTENSION_NAME)
                gd.support_flags -= {.Window}
            }
        }
        if .Raytracing in gd.support_flags {
            gd.support_flags += {.Raytracing}
            rt_exts : []string = {
                vk.KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                vk.KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                vk.KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                vk.KHR_RAY_QUERY_EXTENSION_NAME,
            }
            for ext in rt_exts {
                if !string_contained(supported_extensions[:], ext) {
                    log.errorf("Requested raytracing support but %v was not found", ext)
                    gd.support_flags -= {.Raytracing}
                    break
                }
            }

            if .Raytracing in gd.support_flags {
                for ext in rt_exts {
                    append(&final_extensions, strings.clone_to_cstring(ext, context.temp_allocator))
                }
            }
        }

        // Create logical device
        create_info := vk.DeviceCreateInfo {
            sType = .DEVICE_CREATE_INFO,
            pNext = &features,
            flags = nil,
            queueCreateInfoCount = queue_count,
            pQueueCreateInfos = raw_data(&queue_create_infos),
            enabledExtensionCount = u32(len(final_extensions)),
            ppEnabledExtensionNames = raw_data(final_extensions),
            ppEnabledLayerNames = nil,
            pEnabledFeatures = nil
        }

        r := vk.CreateDevice(gd.physical_device, &create_info, params.allocation_callbacks, &gd.device)
        if r != .SUCCESS {
            log.error("Failed to create device.")
            return gd, r
        }
    }

    // Load proc addrs that come from the device driver
    vk.load_proc_addresses_device(gd.device)

    // Initialize the Vulkan Memory Allocator
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
            flags = {.Externally_Synchronized,.Buffer_Device_Address,.Ext_Memory_Budget},
            physical_device = gd.physical_device,
            device = gd.device,
            preferred_large_heap_block_size = 0,
            allocation_callbacks = params.allocation_callbacks,
            device_memory_callbacks = nil,
            heap_size_limit = nil,
            vulkan_functions = &fns,
            instance = gd.instance,
            vulkan_api_version = api_version_int,
            type_external_memory_handle_types = nil            
        }
        if vma.create_allocator(&info, &gd.allocator) != .SUCCESS {
            log.error("Failed to initialize VMA.")
        }
    }

    // Cache individual device queues
    // We only use one queue from each family
    {
        vk.GetDeviceQueue(gd.device, gd.gfx_queue_family, 0, &gd.gfx_queue)
        vk.GetDeviceQueue(gd.device, gd.compute_queue_family, 0, &gd.compute_queue)
        vk.GetDeviceQueue(gd.device, gd.transfer_queue_family, 0, &gd.transfer_queue)

        // Debug names
        assign_debug_name(gd.device, .QUEUE, u64(uintptr(gd.gfx_queue)), "GFX Queue")

        if gd.compute_queue != gd.gfx_queue {
            assign_debug_name(gd.device, .QUEUE, u64(uintptr(gd.compute_queue)), "Compute Queue")
        }

        if gd.transfer_queue != gd.gfx_queue {
            assign_debug_name(gd.device, .QUEUE, u64(uintptr(gd.transfer_queue)), "Transfer Queue")
        }
    }

    // Create command pools
    {
        gfx_pool_info := vk.CommandPoolCreateInfo {
            sType = .COMMAND_POOL_CREATE_INFO,
            pNext = nil,
            flags = {vk.CommandPoolCreateFlag.TRANSIENT, vk.CommandPoolCreateFlag.RESET_COMMAND_BUFFER},
            queueFamilyIndex = gd.gfx_queue_family
        }
        if vk.CreateCommandPool(gd.device, &gfx_pool_info, params.allocation_callbacks, &gd.gfx_command_pool) != .SUCCESS {
            log.error("Failed to create gfx command pool")
        }
        assign_debug_name(gd.device, .COMMAND_POOL, u64(gd.gfx_command_pool), "GFX Command Pool")

        compute_pool_info := vk.CommandPoolCreateInfo {
            sType = .COMMAND_POOL_CREATE_INFO,
            pNext = nil,
            flags = {vk.CommandPoolCreateFlag.TRANSIENT, vk.CommandPoolCreateFlag.RESET_COMMAND_BUFFER},
            queueFamilyIndex = gd.compute_queue_family
        }
        if vk.CreateCommandPool(gd.device, &compute_pool_info, params.allocation_callbacks, &gd.compute_command_pool) != .SUCCESS {
            log.error("Failed to create compute command pool")
        }
        assign_debug_name(gd.device, .COMMAND_POOL, u64(gd.compute_command_pool), "Compute Command Pool")

        transfer_pool_info := vk.CommandPoolCreateInfo {
            sType = .COMMAND_POOL_CREATE_INFO,
            pNext = nil,
            flags = {vk.CommandPoolCreateFlag.TRANSIENT, vk.CommandPoolCreateFlag.RESET_COMMAND_BUFFER},
            queueFamilyIndex = gd.transfer_queue_family
        }
        if vk.CreateCommandPool(gd.device, &transfer_pool_info, params.allocation_callbacks, &gd.transfer_command_pool) != .SUCCESS {
            log.error("Failed to create transfer command pool")
        }
        assign_debug_name(gd.device, .COMMAND_POOL, u64(gd.transfer_command_pool), "Transfer Command Pool")
    }

    // Create command buffers
    gd.gfx_command_buffers = make([dynamic]vk.CommandBuffer, params.frames_in_flight, context.allocator)
    gd.compute_command_buffers = make([dynamic]vk.CommandBuffer, params.frames_in_flight, context.allocator)
    gd.transfer_command_buffers = make([dynamic]vk.CommandBuffer, params.frames_in_flight, context.allocator)
    {
        gfx_info := vk.CommandBufferAllocateInfo {
            sType = .COMMAND_BUFFER_ALLOCATE_INFO,
            pNext = nil,
            commandPool = gd.gfx_command_pool,
            level = .PRIMARY,
            commandBufferCount = params.frames_in_flight
        }
        if vk.AllocateCommandBuffers(gd.device, &gfx_info, &gd.gfx_command_buffers[0]) != .SUCCESS {
            log.error("Failed to create gfx command buffers")
        }
        compute_info := vk.CommandBufferAllocateInfo {
            sType = .COMMAND_BUFFER_ALLOCATE_INFO,
            pNext = nil,
            commandPool = gd.compute_command_pool,
            level = .PRIMARY,
            commandBufferCount = params.frames_in_flight
        }
        if vk.AllocateCommandBuffers(gd.device, &compute_info, &gd.compute_command_buffers[0]) != .SUCCESS {
            log.error("Failed to create compute command buffers")
        }
        transfer_info := vk.CommandBufferAllocateInfo {
            sType = .COMMAND_BUFFER_ALLOCATE_INFO,
            pNext = nil,
            commandPool = gd.transfer_command_pool,
            level = .PRIMARY,
            commandBufferCount = params.frames_in_flight
        }
        if vk.AllocateCommandBuffers(gd.device, &transfer_info, &gd.transfer_command_buffers[0]) != .SUCCESS {
            log.error("Failed to create transfer command buffers")
        }

        // Debug naming
        sb: strings.Builder
        strings.builder_init(&sb, allocator = context.temp_allocator)
        for i in 0..<params.frames_in_flight {
            fmt.sbprintf(&sb, "GFX Command Buffer #%v", i)
            c_name, _ := strings.to_cstring(&sb)
            assign_debug_name(gd.device, .COMMAND_BUFFER, u64(uintptr(gd.gfx_command_buffers[i])), c_name)
            strings.builder_reset(&sb)
        }
        for i in 0..<params.frames_in_flight {
            fmt.sbprintf(&sb, "Compute Command Buffer #%v", i)
            c_name, _ := strings.to_cstring(&sb)
            assign_debug_name(gd.device, .COMMAND_BUFFER, u64(uintptr(gd.compute_command_buffers[i])), c_name)
            strings.builder_reset(&sb)
        }
        for i in 0..<params.frames_in_flight {
            fmt.sbprintf(&sb, "Transfer Command Buffer #%v", i)
            c_name, _ := strings.to_cstring(&sb)
            assign_debug_name(gd.device, .COMMAND_BUFFER, u64(uintptr(gd.transfer_command_buffers[i])), c_name)
            strings.builder_reset(&sb)
        }
    }

    // Create bindless descriptor set
    samplers: [TOTAL_IMMUTABLE_SAMPLERS]vk.Sampler
    {
        // Create immutable samplers
        {
            sampler_infos := make([dynamic]vk.SamplerCreateInfo, len = 0, cap = TOTAL_IMMUTABLE_SAMPLERS, allocator = context.temp_allocator)
            append(&sampler_infos, vk.SamplerCreateInfo {
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
            })
            append(&sampler_infos, vk.SamplerCreateInfo {
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
            })
            append(&sampler_infos, vk.SamplerCreateInfo {
                sType = .SAMPLER_CREATE_INFO,
                pNext = nil,
                flags = nil,
                magFilter = .LINEAR,
                minFilter = .LINEAR,
                mipmapMode = .NEAREST,
                addressModeU = .CLAMP_TO_EDGE,
                addressModeV = .CLAMP_TO_EDGE,
                addressModeW = .CLAMP_TO_EDGE,
                mipLodBias = 0.0,
                anisotropyEnable = true,
                maxAnisotropy = 16.0
            })

            debug_names : [Immutable_Sampler_Index]cstring = {
                .Aniso16 = "Anisotropic filtering 16x Sampler",
                .Point = "Point Sampler",
                .PostFX = "PostFX Sampler",
            }

            for &s, i in sampler_infos {
                vk.CreateSampler(gd.device, &s, params.allocation_callbacks, &samplers[i])
                assign_debug_name(gd.device, .SAMPLER, u64(samplers[i]), debug_names[Immutable_Sampler_Index(i)])
            }
        }

        image_binding := vk.DescriptorSetLayoutBinding {
            binding = u32(Bindless_Descriptor_Bindings.Images),
            descriptorType = .SAMPLED_IMAGE,
            descriptorCount = MAXIMUM_BINDLESS_IMAGES,
            stageFlags = {.FRAGMENT},
            pImmutableSamplers = nil
        }
        sampler_binding := vk.DescriptorSetLayoutBinding {
            binding = u32(Bindless_Descriptor_Bindings.Samplers),
            descriptorType = .SAMPLER,
            descriptorCount = TOTAL_IMMUTABLE_SAMPLERS,
            stageFlags = {.FRAGMENT},
            pImmutableSamplers = raw_data(samplers[:])
        }
        AS_binding := vk.DescriptorSetLayoutBinding {
            binding = u32(Bindless_Descriptor_Bindings.AccelerationStructures),
            descriptorType = .ACCELERATION_STRUCTURE_KHR,
            descriptorCount = TOTAL_AS_DESCRIPTORS,
            stageFlags = {.FRAGMENT},
            pImmutableSamplers = nil
        }
        bindings : []vk.DescriptorSetLayoutBinding = {image_binding, sampler_binding, AS_binding}
        bindings_count : u32 = 2
        if .Raytracing in gd.support_flags {
            bindings_count += 1
        }

        binding_flags : vk.DescriptorBindingFlags = {.PARTIALLY_BOUND,.UPDATE_AFTER_BIND}
        binding_flags_plural : []vk.DescriptorBindingFlags = {binding_flags,binding_flags,binding_flags}
        binding_flags_info := vk.DescriptorSetLayoutBindingFlagsCreateInfo {
            sType = .DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            pNext = nil,
            bindingCount = bindings_count,
            pBindingFlags = &binding_flags_plural[0]
        }
        layout_info := vk.DescriptorSetLayoutCreateInfo {
            sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext = &binding_flags_info,
            flags = {.UPDATE_AFTER_BIND_POOL},
            bindingCount = bindings_count,
            pBindings = raw_data(bindings[:])
        }
        if vk.CreateDescriptorSetLayout(gd.device, &layout_info, params.allocation_callbacks, &gd.descriptor_set_layout) != .SUCCESS {
            log.error("Failed to create static descriptor set layout.")
        }

        sizes : []vk.DescriptorPoolSize = {
            vk.DescriptorPoolSize {
                type = .SAMPLED_IMAGE,
                descriptorCount = MAXIMUM_BINDLESS_IMAGES
            },
            vk.DescriptorPoolSize {
                type = .SAMPLER,
                descriptorCount = TOTAL_IMMUTABLE_SAMPLERS
            },
            vk.DescriptorPoolSize {
                type = .ACCELERATION_STRUCTURE_KHR,
                descriptorCount = TOTAL_AS_DESCRIPTORS
            }
        }

        pool_info := vk.DescriptorPoolCreateInfo {
            sType = .DESCRIPTOR_POOL_CREATE_INFO,
            pNext = nil,
            flags = {.UPDATE_AFTER_BIND},
            maxSets = 1,
            poolSizeCount = bindings_count,
            pPoolSizes = raw_data(sizes[:])
        }
        if vk.CreateDescriptorPool(gd.device, &pool_info, params.allocation_callbacks, &gd.descriptor_pool) != .SUCCESS {
            log.error("Failed to create descriptor pool.")
        }

        set_info := vk.DescriptorSetAllocateInfo {
            sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
            pNext = nil,
            descriptorPool = gd.descriptor_pool,
            descriptorSetCount = 1,
            pSetLayouts = &gd.descriptor_set_layout
        }
        if vk.AllocateDescriptorSets(gd.device, &set_info, &gd.descriptor_set) != .SUCCESS {
            log.error("Failed to allocate descriptor sets.")
        }
    }

    // Create global pipeline layout
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
            pSetLayouts = &gd.descriptor_set_layout,
            pushConstantRangeCount = 1,
            pPushConstantRanges = &pc_range
        }
        if vk.CreatePipelineLayout(gd.device, &layout_info, params.allocation_callbacks, &gd.gfx_pipeline_layout) != .SUCCESS {
            log.error("Failed to create graphics pipeline layout.")
        }

        pc_range = vk.PushConstantRange {
            stageFlags = {.COMPUTE},
            offset = 0,
            size = PUSH_CONSTANTS_SIZE
        }
        layout_info = vk.PipelineLayoutCreateInfo {
            sType = .PIPELINE_LAYOUT_CREATE_INFO,
            pNext = nil,
            flags = nil,
            setLayoutCount = 1,
            pSetLayouts = &gd.descriptor_set_layout,
            pushConstantRangeCount = 1,
            pPushConstantRanges = &pc_range
        }
        if vk.CreatePipelineLayout(gd.device, &layout_info, params.allocation_callbacks, &gd.compute_pipeline_layout) != .SUCCESS {
            log.error("Failed to create graphics pipeline layout.")
        }
    }

    // Init Handle_Maps
    {
        hm.init(&gd.buffers)
        hm.init(&gd.images)
        hm.init(&gd.acceleration_structures)
        hm.init(&gd.semaphores)
        hm.init(&gd.pipelines)
    }

    // Init queues
    {
        queue.init(&gd.pending_images)
        queue.init(&gd.buffer_deletes)
        queue.init(&gd.image_deletes)
        queue.init(&gd.BLAS_queued_build_infos)
    }

    // Create staging buffer
    {
        info := Buffer_Info {
            size = STAGING_BUFFER_SIZE,
            usage = {.TRANSFER_SRC},
            alloc_flags = {.Mapped},
            required_flags = {.DEVICE_LOCAL,.HOST_VISIBLE,.HOST_COHERENT},
            name = "Global staging buffer",
        }
        gd.staging_buffer = create_buffer(&gd, &info)
    }

    // Create acceleration structure buffers
    if .Raytracing in gd.support_flags {
        info := Buffer_Info {
            size = AS_BUFFER_SIZE,
            usage = {.ACCELERATION_STRUCTURE_STORAGE_KHR},
            required_flags = {.DEVICE_LOCAL},
            name = "Global acceleration structure buffer",
        }
        gd.AS_buffer = create_buffer(&gd, &info)

        info.size = TLAS_INSTANCE_BUFFER_SIZE
        info.usage = {.ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,.TRANSFER_DST}
        info.required_flags += {.HOST_COHERENT,.HOST_VISIBLE}
        info.alloc_flags = {.Mapped}
        gd.TLAS_instance_buffer = create_buffer(&gd, &info)

        // Set this handle to invalid so that it will be lazily allocated
        gd.AS_scratch_buffer.generation = 0xFFFFFFFF

        // Put dummy acceleration in slot 0 so that {index = 0, generation = 0}
        // Will always be a null AS
        unused := hm.insert(&gd.acceleration_structures, AccelerationStructure {
            handle = 0,
            offset = 0
        })
    }

    // Create transfer timeline semaphore
    {
        info := Semaphore_Info {
            type = .TIMELINE,
            init_value = 0,
            name = "Transfer timeline"
        }
        gd.transfer_timeline = create_semaphore(&gd, &info)
    }

    // Put null texture in slot 0
    {
        // Make black and purple texture
        size :: 1 << 4
        tex_data: [size*size*4]u8
        for i := 0; i < len(tex_data); i += 4 {
            pixel_idx := i/4
            is_odd : int = int((pixel_idx / 16) % 2 == 1)
            result: u8 = 0xFF if (pixel_idx) % 2 == is_odd else 0x00
            tex_data[i] = result
            tex_data[i + 1] = 0x0
            tex_data[i + 2] = result
            tex_data[i + 3] = 0xFF
        }
        
        info := Image_Create {
            flags = nil,
            image_type = .D2,
            format = .R8G8B8A8_UNORM,
            extent = {
                width = size,
                height = size,
                depth = 1
            },
            has_mipmaps = false,
            array_layers = 1,
            samples = {._1},
            tiling = .OPTIMAL,
            usage = {.SAMPLED},
            alloc_flags = nil,
            name = "Null image"
        }
        handle, ok := sync_create_image_with_data(&gd, &info, tex_data[:])
        if !ok {
            log.error("Error creating null texture")
        }
    }

    // Initialize pipeline cache
    {
        // Check for existance of existing cache
        pipeline_cache_bytes, cache_exists := os.read_entire_file_from_filename(PIPELINE_CACHE_FILENAME)
        data_ptr: rawptr
        data_size: int
        if cache_exists {
            data_ptr = &pipeline_cache_bytes[0]
            data_size = len(pipeline_cache_bytes)
        } else {
            log.warn("Didn't find shader cache.")
        }

        info := vk.PipelineCacheCreateInfo {
            sType = .PIPELINE_CACHE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            initialDataSize = data_size,
            pInitialData = data_ptr
        }
        res := vk.CreatePipelineCache(gd.device, &info, gd.alloc_callbacks, &gd.pipeline_cache)
        if res != nil {
            log.errorf("Error creating pipeline cache: %v", res)
        }
    }

    return gd, .SUCCESS
}

quit_vulkan :: proc(gd: ^Graphics_Device) {
    // Save the pipeline cache to a file
    {
        res: vk.Result
        data_size: int
        res = vk.GetPipelineCacheData(gd.device, gd.pipeline_cache, &data_size, nil)
        if res != nil {
            log.errorf("Error getting pipeline cache size: %v", res)
        }
        assert(data_size > 0)

        pipeline_data := make([dynamic]byte, data_size, context.temp_allocator)
        res = vk.GetPipelineCacheData(gd.device, gd.pipeline_cache, &data_size, &pipeline_data[0])
        if res != nil {
            log.errorf("Error getting pipeline cache data: %v", res)
        }

        cache_file, err := create_write_file(PIPELINE_CACHE_FILENAME)
        if err == nil {
            os.write(cache_file, pipeline_data[:])
        } else {
            log.errorf("Error writing pipeline cache file: %v", err)
        }
        os.close(cache_file)

        vk.DestroyPipelineCache(gd.device, gd.pipeline_cache, gd.alloc_callbacks)
    }

    vk.DestroySwapchainKHR(gd.device, gd.swapchain, gd.alloc_callbacks)
    vk.DestroySurfaceKHR(gd.instance, gd.surface, gd.alloc_callbacks)

    for buffer in gd.buffers.values {
        vma.destroy_buffer(gd.allocator, buffer.buffer, buffer.allocation)
    }

    // @TODO: Why does this cause a null-pointer dereference?
    // for image in gd.images.values {
    //     vk.DestroyImageView(gd.device, image.image_view, gd.alloc_callbacks)
    //     vma.destroy_image(gd.allocator, image.image, image.allocation)
    // }

    for semaphore in gd.semaphores.values {
        vk.DestroySemaphore(gd.device, semaphore, gd.alloc_callbacks)
    }

    for pipeline in gd.pipelines.values {
        vk.DestroyPipeline(gd.device, pipeline, gd.alloc_callbacks)
    }

    delete(gd.acquire_semaphores)
    delete(gd.present_semaphores)
    delete(gd.swapchain_images)
    delete(gd.gfx_command_buffers)
    delete(gd.compute_command_buffers)
    delete(gd.transfer_command_buffers)
    queue.destroy(&gd.pending_images)
    queue.destroy(&gd.buffer_deletes)
    queue.destroy(&gd.image_deletes)
    hm.destroy(&gd.buffers)
    hm.destroy(&gd.images)
    hm.destroy(&gd.semaphores)
    hm.destroy(&gd.pipelines)

    vma.destroy_allocator(gd.allocator)

    for sampler in gd.immutable_samplers {
        vk.DestroySampler(gd.device, sampler, gd.alloc_callbacks)
    }

    vk.DestroyPipelineLayout(gd.device, gd.gfx_pipeline_layout, gd.alloc_callbacks)
    vk.DestroyPipelineLayout(gd.device, gd.compute_pipeline_layout, gd.alloc_callbacks)
    vk.DestroyDescriptorPool(gd.device, gd.descriptor_pool, gd.alloc_callbacks)
    vk.DestroyCommandPool(gd.device, gd.gfx_command_pool, gd.alloc_callbacks)
    vk.DestroyCommandPool(gd.device, gd.compute_command_pool, gd.alloc_callbacks)
    vk.DestroyCommandPool(gd.device, gd.transfer_command_pool, gd.alloc_callbacks)
    vk.DestroyDevice(gd.device, gd.alloc_callbacks)
    vk.DestroyInstance(gd.instance, gd.alloc_callbacks)
}

device_wait_idle :: proc(gd: ^Graphics_Device) -> vk.Result {
    return vk.DeviceWaitIdle(gd.device)
}

assign_debug_name :: proc(device: vk.Device, object_type: vk.ObjectType, object_handle: u64, name: cstring) {
    name_info := vk.DebugUtilsObjectNameInfoEXT {
        sType = .DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
        pNext = nil,
        objectType = object_type,
        objectHandle = object_handle,
        pObjectName = name
    }
    r := vk.SetDebugUtilsObjectNameEXT(device, &name_info)
    if r != .SUCCESS {
        log.errorf("Failed to set object's debug name: %v", r)
    }
}

// @TODO: Allow more configurability of swapchain options
// particularly pertaining to presentation mode and image format
SwapchainInfo :: struct {
    dimensions: [2]uint,
    present_mode: vk.PresentModeKHR
}

window_create_swapchain :: proc(gd: ^Graphics_Device, info: SwapchainInfo) -> bool {
    surface_caps: vk.SurfaceCapabilitiesKHR
    vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(gd.physical_device, gd.surface, &surface_caps)

    dimensions := [2]u32{u32(info.dimensions.x), u32(info.dimensions.y)}
    if dimensions.x < surface_caps.minImageExtent.width {
        dimensions.x = surface_caps.minImageExtent.width
    }
    if dimensions.x > surface_caps.maxImageExtent.width {
        dimensions.x = surface_caps.maxImageExtent.width
    }
    if dimensions.y < surface_caps.minImageExtent.height {
        dimensions.y = surface_caps.minImageExtent.height
    }
    if dimensions.y > surface_caps.maxImageExtent.height {
        dimensions.y = surface_caps.maxImageExtent.height
    }

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
            width = dimensions.x,
            height = dimensions.y
        },
        imageArrayLayers = 1,
        imageUsage = {.COLOR_ATTACHMENT},
        imageSharingMode = .EXCLUSIVE,
        queueFamilyIndexCount = 1,
        pQueueFamilyIndices = &gd.gfx_queue_family,
        preTransform = {.IDENTITY},
        compositeAlpha = {.OPAQUE},
        presentMode = info.present_mode,
        clipped = true,
        oldSwapchain = gd.swapchain
    }
    temp: vk.SwapchainKHR
    if vk.CreateSwapchainKHR(gd.device, &create_info, gd.alloc_callbacks, &temp) != .SUCCESS {
        return false
    }

    vk.DestroySwapchainKHR(gd.device, gd.swapchain, gd.alloc_callbacks)
    gd.swapchain = temp

    // Get swapchain images
    swapchain_images: [dynamic]vk.Image
    image_count : u32 = 0
    {
        vk.GetSwapchainImagesKHR(gd.device, gd.swapchain, &image_count, nil)
        swapchain_images = make([dynamic]vk.Image, image_count, context.temp_allocator)
        vk.GetSwapchainImagesKHR(gd.device, gd.swapchain, &image_count, raw_data(swapchain_images))
    }

    swapchain_image_views := make([dynamic]vk.ImageView, image_count, context.temp_allocator)
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

    sb: strings.Builder
    strings.builder_init(&sb, allocator = context.temp_allocator)
    {
        clear(&gd.acquire_semaphores)
        clear(&gd.present_semaphores)
        gd := gd
        resize(&gd.acquire_semaphores, image_count)
        resize(&gd.present_semaphores, image_count)
        resize(&gd.swapchain_images, image_count)
        for i : u32 = 0; i < image_count; i += 1 {
            im := Image {
                image = swapchain_images[i],
                image_view = swapchain_image_views[i]
            }
            gd.swapchain_images[i] = Texture_Handle(hm.insert(&gd.images, im))

            sem_name := fmt.sbprintf(&sb, "Acquire binary #%v", i)
            info := Semaphore_Info {
                type = .BINARY,
                name = strings.unsafe_string_to_cstring(sem_name)
            }
            gd.acquire_semaphores[i] = create_semaphore(gd, &info)
            strings.builder_reset(&sb)


            sem_name = fmt.sbprintf(&sb, "Present binary #%v", i)
            info.name = strings.unsafe_string_to_cstring(sem_name)
            gd.present_semaphores[i] = create_semaphore(gd, &info)
            strings.builder_reset(&sb)
        }
    }

    return true
}

init_sdl2_window :: proc(gd: ^Graphics_Device, window: ^sdl2.Window, mode: vk.PresentModeKHR) -> bool {
    sdl2.Vulkan_CreateSurface(window, gd.instance, &gd.surface) or_return

    width, height : i32 = 0, 0
    sdl2.Vulkan_GetDrawableSize(window, &width, &height)

    info := SwapchainInfo {
        dimensions = {uint(width), uint(height)},
        present_mode = mode
    }
    window_create_swapchain(gd, info) or_return

    return true
}

resize_window :: proc(gd: ^Graphics_Device, info: SwapchainInfo) -> bool {
    // The graphics device's swapchain should exist
    assert(gd.swapchain != 0)

    old_swapchain_handles := make([dynamic]Texture_Handle, len(gd.swapchain_images), context.temp_allocator)
    for handle, i in gd.swapchain_images {
        old_swapchain_handles[i] = handle
    }

    window_create_swapchain(gd, info) or_return

    // Remove stale swapchain image handles
    for handle in old_swapchain_handles {
        hm.remove(&gd.images, hm.Handle(handle))
    }

    return true
}

in_flight_idx :: proc(gd: ^Graphics_Device) -> u64 {
    return gd.frame_count % u64(gd.frames_in_flight)
}

Buffer_Info :: struct {
    size: vk.DeviceSize,
    usage: vk.BufferUsageFlags,
    alloc_flags: vma.Allocation_Create_Flags,
    // queue_family: Queue_Family,
    required_flags: vk.MemoryPropertyFlags,
    name: string,
}

Buffer :: struct {
    buffer: vk.Buffer,
    address: vk.DeviceAddress,
    allocation: vma.Allocation,
    alloc_info: vma.Allocation_Info,
}

Buffer_Delete :: struct {
    death_frame: u64,
    buffer: vk.Buffer,
    allocation: vma.Allocation,
}

create_buffer :: proc(gd: ^Graphics_Device, buf_info: ^Buffer_Info) -> Buffer_Handle {
    qfis : []u32 = {gd.gfx_queue_family,gd.compute_queue_family,gd.transfer_queue_family}

    info := vk.BufferCreateInfo {
        sType = .BUFFER_CREATE_INFO,
        pNext = nil,
        flags = nil,
        size = buf_info.size,
        usage = buf_info.usage + {.SHADER_DEVICE_ADDRESS},
        sharingMode = .CONCURRENT,
        queueFamilyIndexCount = u32(len(qfis)),
        pQueueFamilyIndices = raw_data(qfis[:])
    }
    alloc_info := vma.Allocation_Create_Info {
        flags = buf_info.alloc_flags,
        usage = .Auto,
        required_flags = buf_info.required_flags,
        preferred_flags = nil,
        priority = 1.0
    }
    b: Buffer
    r := vma.create_buffer(gd.allocator, &info, &alloc_info, &b.buffer, &b.allocation, &b.alloc_info)
    if r != .SUCCESS {
        log.errorf("Failed to create buffer: %v", r)
    }

    bda_info := vk.BufferDeviceAddressInfo {
        sType = .BUFFER_DEVICE_ADDRESS_INFO,
        pNext = nil,
        buffer = b.buffer
    }
    b.address = vk.GetBufferDeviceAddress(gd.device, &bda_info)

    if len(buf_info.name) > 0 {
        assign_debug_name(gd.device, .BUFFER, u64(b.buffer), strings.unsafe_string_to_cstring(buf_info.name))
    } else {
        log.warn("Buffer was created without a debug name")
    }

    return Buffer_Handle(hm.insert(&gd.buffers, b))
}

get_buffer :: proc(gd: ^Graphics_Device, handle: Buffer_Handle) -> (^Buffer, bool) {
    return hm.get(&gd.buffers, hm.Handle(handle))
}

// Blocking function for writing to a GPU buffer
// Use when the data is small or if you don't care about stalling
// @TODO: Maybe add sync_write_buffers()?
sync_write_buffer :: proc(
    gd: ^Graphics_Device,
    out_buffer: Buffer_Handle,
    in_slice: []$Element_Type,
    base_offset : u32 = 0,
    sync: SyncInfo = {}
) -> bool {
    if len(in_slice) == 0 {
        return false
    }

    cb := gd.transfer_command_buffers[in_flight_idx(gd)]
    out_buf := get_buffer(gd, out_buffer) or_return
    in_bytes : []byte = slice.from_ptr(cast([^]byte)slice.as_ptr(in_slice), len(in_slice) * size_of(Element_Type))
    base_offset_bytes := base_offset * size_of(Element_Type)

    total_bytes := len(in_bytes)
    bytes_transferred := 0

    if out_buf.alloc_info.mapped_data != nil {
        in_p := &in_bytes[0]
        out_p := rawptr(uintptr(out_buf.alloc_info.mapped_data) + uintptr(base_offset_bytes))
        mem.copy(out_p, in_p, len(in_bytes))
    } else {

        // Staging buffer
        sb := get_buffer(gd, gd.staging_buffer) or_return
        sb_ptr := sb.alloc_info.mapped_data

        for bytes_transferred < total_bytes {
            iter_size := min(total_bytes - bytes_transferred, STAGING_BUFFER_SIZE)

            // Copy to staging buffer
            p := &in_bytes[bytes_transferred]
            mem.copy(sb_ptr, p, iter_size)

            // Begin transfer command buffer
            begin_info := vk.CommandBufferBeginInfo {
                sType = .COMMAND_BUFFER_BEGIN_INFO,
                pNext = nil,
                flags = {.ONE_TIME_SUBMIT},
                pInheritanceInfo = nil
            }
            vk.BeginCommandBuffer(cb, &begin_info)

            // Copy from staging buffer to actual buffer
            buffer_copy := vk.BufferCopy {
                srcOffset = 0,
                dstOffset = vk.DeviceSize(bytes_transferred) + vk.DeviceSize(base_offset_bytes),
                size = vk.DeviceSize(iter_size)
            }
            vk.CmdCopyBuffer(cb, sb.buffer, out_buf.buffer, 1, &buffer_copy)

            vk.EndCommandBuffer(cb)

            // Actually submit this lonesome command to the transfer queue
            cb_info := vk.CommandBufferSubmitInfo {
                sType = .COMMAND_BUFFER_SUBMIT_INFO,
                pNext = nil,
                commandBuffer = cb,
                deviceMask = 0
            }

            // Honor sync info
            wait_infos := make([dynamic]vk.SemaphoreSubmitInfoKHR, len(sync.wait_ops), allocator = context.temp_allocator)
            signal_infos := make([dynamic]vk.SemaphoreSubmitInfoKHR, len(sync.signal_ops), allocator = context.temp_allocator)
            {
                build_submit_infos(gd, &wait_infos, sync.wait_ops)
                build_submit_infos(gd, &signal_infos, sync.signal_ops)

                // Increment transfer timeline counter
                semaphore := hm.get(&gd.semaphores, hm.Handle(gd.transfer_timeline)) or_return
                new_timeline_value := gd.transfers_completed + 1
                signal_info := vk.SemaphoreSubmitInfo {
                    sType = .SEMAPHORE_SUBMIT_INFO,
                    pNext = nil,
                    semaphore = semaphore^,
                    value = new_timeline_value,
                    stageMask = {.ALL_COMMANDS},    // @TODO: This is a bit heavy-handed
                    deviceIndex = 0
                }
                append(&signal_infos, signal_info)
            }

            submit_info := vk.SubmitInfo2 {
                sType = .SUBMIT_INFO_2,
                pNext = nil,
                flags = nil,
                commandBufferInfoCount = 1,
                pCommandBufferInfos = &cb_info,
                waitSemaphoreInfoCount = u32(len(wait_infos)),
                pWaitSemaphoreInfos = raw_data(wait_infos),
                signalSemaphoreInfoCount = u32(len(signal_infos)),
                pSignalSemaphoreInfos = raw_data(signal_infos)
            }
            if vk.QueueSubmit2(gd.transfer_queue, 1, &submit_info, 0) != .SUCCESS {
                log.error("Failed to submit to transfer queue.")
            }

            // CPU wait
            // @TODO: Should be able to get rid of this wait by having
            // gfx queue wait on transfer timeline semaphore
            semaphore := hm.get(&gd.semaphores, hm.Handle(gd.transfer_timeline)) or_return
            new_timeline_value := gd.transfers_completed + 1
            wait_info := vk.SemaphoreWaitInfo {
                sType = .SEMAPHORE_WAIT_INFO,
                pNext = nil,
                flags = nil,
                semaphoreCount = 1,
                pSemaphores = semaphore,
                pValues = &new_timeline_value
            }
            if vk.WaitSemaphores(gd.device, &wait_info, max(u64)) != .SUCCESS {
                log.error("Failed to wait for transfer semaphore.")
            }

            gd.transfers_completed += 1
            bytes_transferred += iter_size
        }
    }


    return true
}

delete_buffer :: proc(gd: ^Graphics_Device, handle: Buffer_Handle) -> bool {
    buffer := get_buffer(gd, handle) or_return

    buffer_delete := Buffer_Delete {
        death_frame = gd.frame_count + u64(gd.frames_in_flight),
        buffer = buffer.buffer,
        allocation = buffer.allocation
    }
    queue.append(&gd.buffer_deletes, buffer_delete)
    hm.remove(&gd.buffers, hm.Handle(handle))

    return true
}

Image_Create :: struct {
    flags: vk.ImageCreateFlags,
    image_type: vk.ImageType,
    format: vk.Format,
    extent: vk.Extent3D,
    has_mipmaps: bool,
    mip_count: u32,
    array_layers: u32,
    samples: vk.SampleCountFlags,
    tiling: vk.ImageTiling,
    usage: vk.ImageUsageFlags,
    alloc_flags: vma.Allocation_Create_Flags,
    name: cstring
}
Image :: struct {
    image: vk.Image,
    image_view: vk.ImageView,
    allocation: vma.Allocation,
    alloc_info: vma.Allocation_Info,
    extent: vk.Extent3D
}
Image_Delete :: struct {
    death_frame: u64,
    image: vk.Image,
    image_view: vk.ImageView,
    allocation: vma.Allocation
}
Pending_Image :: struct {
    handle: Texture_Handle,
    old_layout: vk.ImageLayout,
    new_layout: vk.ImageLayout,
    aspect_mask: vk.ImageAspectFlags,
    src_queue_family: u32,
    array_layers: u32,
    mip_count: u32,
}

vk_format_pixel_size :: proc(format: vk.Format) -> int {
    #partial switch format {
        case .R8G8B8A8_UNORM: return 4
        case .R8G8B8A8_SRGB: return 4
        case .BC7_SRGB_BLOCK: return 1 //each 128-bit (16-byte) compressed texel block encodes a 4×4 rectangle
        case: {
            log.errorf("Tried to get byte size of unsupported pixel format (just add another switch case): %v", format)
        }
    }
    return 0
}

vk_format_block_size :: proc(format: vk.Format) -> int {
    #partial switch format {
        case .BC7_SRGB_BLOCK: return 16
        case .BC7_UNORM_BLOCK: return 16
        case .R8G8B8A8_UNORM: return 1
        case .R8G8B8A8_SRGB: return 1
        case: {
            log.errorf("Unsupported block size format: %v", format)
        }
    }

    return 0
}

create_image :: proc(gd: ^Graphics_Device, image_info: ^Image_Create) -> Texture_Handle {
    // TRANSFER_DST is required for layout transitions period.
    // SAMPLED is required because all images in the system are
    // available in the bindless images array
    image_info.usage += {.SAMPLED,.TRANSFER_DST}

    // Calculate mipmap count
    mip_count : u32 = 1
    if image_info.has_mipmaps {
        // Mipmaps are only for 2D images at least rn
        assert(image_info.image_type == .D2)

        highest_bit_32 :: proc(n: u32) -> u32 {
            n := n
            i : u32 = 0

            for n > 0 {
                n = n >> 1
                i += 1
            }
            return i
        }

        max_dim := max(image_info.extent.width, image_info.extent.height)
        mip_count = highest_bit_32(max_dim)
    }

    image: Image
    {
        info := vk.ImageCreateInfo {
            sType = .IMAGE_CREATE_INFO,
            pNext = nil,
            flags = image_info.flags,
            imageType = image_info.image_type,
            format = image_info.format,
            extent = image_info.extent,
            mipLevels = mip_count,
            arrayLayers = image_info.array_layers,
            samples = image_info.samples,
            tiling = image_info.tiling,
            usage = image_info.usage,
            sharingMode = .EXCLUSIVE,
            queueFamilyIndexCount = 1,
            pQueueFamilyIndices = &gd.gfx_queue_family,
            initialLayout = .UNDEFINED
        }
        alloc_info := vma.Allocation_Create_Info {
            flags = image_info.alloc_flags,
            usage = .Auto,
            required_flags = {.DEVICE_LOCAL},
            preferred_flags = nil,
            priority = 1.0
        }
        if vma.create_image(
            gd.allocator,
            &info,
            &alloc_info,
            &image.image,
            &image.allocation,
            &image.alloc_info
        ) != .SUCCESS {
            log.error("Failed to create image.")
        }
    }
    image.extent = image_info.extent

    // Create the image view
    {
        // If image_info.flags contains .CUBE_COMPATIBLE then we know we're making a cubemap
        view_type: vk.ImageViewType
        if .CUBE_COMPATIBLE in image_info.flags {
            view_type = .CUBE
        } else {
            switch image_info.image_type {
                case .D1: view_type = .D1
                case .D2: view_type = .D2
                case .D3: view_type = .D3
            }
        }

        aspect_mask : vk.ImageAspectFlags = {.COLOR}
        if image_info.format == .D32_SFLOAT {
            aspect_mask = {.DEPTH}
        }

        subresource_range := vk.ImageSubresourceRange {
            aspectMask = aspect_mask,
            baseMipLevel = 0,
            levelCount = u32(mip_count),
            baseArrayLayer = 0,
            layerCount = image_info.array_layers
        }

        info := vk.ImageViewCreateInfo {
            sType = .IMAGE_VIEW_CREATE_INFO,
            pNext = nil,
            flags = nil,
            image = image.image,
            viewType = view_type,
            format = image_info.format,
            components = IDENTITY_COMPONENT_SWIZZLE,
            subresourceRange = subresource_range
        }
        if vk.CreateImageView(gd.device, &info, gd.alloc_callbacks, &image.image_view) != .SUCCESS {
            log.error("Failed to create image view.")
        }
    }

    // Set debug names
    if len(image_info.name) > 0 {
        sb: strings.Builder
        strings.builder_init(&sb, allocator = context.temp_allocator)
        image_name := fmt.sbprintf(&sb, "%v image", image_info.name)
        assign_debug_name(gd.device, .IMAGE, u64(image.image), strings.unsafe_string_to_cstring(image_name))
        strings.builder_reset(&sb)

        view_name := fmt.sbprintf(&sb, "%v image view", image_info.name)
        assign_debug_name(gd.device, .IMAGE_VIEW, u64(image.image_view), strings.unsafe_string_to_cstring(view_name))
        strings.builder_reset(&sb)
    } else {
        log.warn("Creating image with no debug name!")
    }

    return Texture_Handle(hm.insert(&gd.images, image))
}

new_bindless_image :: proc(gd: ^Graphics_Device, info: ^Image_Create, layout: vk.ImageLayout) -> Texture_Handle {
    handle := create_image(gd, info)
    image, ok := hm.get(&gd.images, hm.Handle(handle))
    if !ok {
        log.error("Error in new_bindless_image()")
    }

    aspect_mask : vk.ImageAspectFlags = {.COLOR}
    if info.format == .D32_SFLOAT {
        aspect_mask = {.DEPTH}
    }

    // Calculate mipmap count
    mip_count : u32 = 1
    if info.has_mipmaps {
        // Mipmaps are only for 2D images at least rn
        assert(info.image_type == .D2)

        highest_bit_32 :: proc(n: u32) -> u32 {
            n := n
            i : u32 = 0

            for n > 0 {
                n = n >> 1
                i += 1
            }
            return i
        }

        max_dim := max(info.extent.width, info.extent.height)
        mip_count = highest_bit_32(max_dim)
    }


    queue.push_back(&gd.pending_images, Pending_Image {
        handle = handle,
        old_layout = .UNDEFINED,
        new_layout = layout,
        aspect_mask = aspect_mask,
        src_queue_family = gd.gfx_queue_family,
        array_layers = info.array_layers,
        mip_count = mip_count,
    })



    // Record queue family ownership transfer
    // @TODO: Barrier is overly opinionated about future usage
    cb_idx := CommandBuffer_Index(in_flight_idx(gd))
    cb := gd.transfer_command_buffers[cb_idx]

    vk.BeginCommandBuffer(cb, &vk.CommandBufferBeginInfo {
        sType = .COMMAND_BUFFER_BEGIN_INFO,
        pNext = nil,
        flags = {.ONE_TIME_SUBMIT},
        pInheritanceInfo = nil
    })

    semaphore, ok2 := hm.get(&gd.semaphores, hm.Handle(gd.transfer_timeline))
    if !ok2 {
        log.error("Error getting transfer timeline semaphore")
    }

    barriers := []Image_Barrier {
        {
            src_stage_mask = {.ALL_COMMANDS},
            src_access_mask = {.MEMORY_READ,.MEMORY_WRITE},
            dst_stage_mask = {.ALL_COMMANDS},                                 //Ignored during release operation
            dst_access_mask = {.MEMORY_READ,.MEMORY_WRITE},                   //Ignored during release operation
            old_layout = .UNDEFINED,
            new_layout = layout,
            src_queue_family = gd.transfer_queue_family,
            dst_queue_family = gd.gfx_queue_family,
            image = image.image,
            subresource_range = {
                aspectMask = aspect_mask,
                baseMipLevel = 0,
                levelCount = 1,
                baseArrayLayer = 0,
                layerCount = info.array_layers
            }
        }
    }
    cmd_transfer_pipeline_barriers(gd, cb_idx, {}, barriers)

    vk.EndCommandBuffer(cb)

    // Actually submit this lonesome command to the transfer queue
    cb_info := vk.CommandBufferSubmitInfo {
        sType = .COMMAND_BUFFER_SUBMIT_INFO,
        pNext = nil,
        commandBuffer = cb,
        deviceMask = 0
    }

    // Increment transfer timeline counter
    new_timeline_value := gd.transfers_completed + 1
    signal_info := vk.SemaphoreSubmitInfo {
        sType = .SEMAPHORE_SUBMIT_INFO,
        pNext = nil,
        semaphore = semaphore^,
        value = new_timeline_value,
        stageMask = {.ALL_COMMANDS},    // @TODO: This is a bit heavy-handed
        deviceIndex = 0
    }

    submit_info := vk.SubmitInfo2 {
        sType = .SUBMIT_INFO_2,
        pNext = nil,
        flags = nil,
        commandBufferInfoCount = 1,
        pCommandBufferInfos = &cb_info,
        signalSemaphoreInfoCount = 1,
        pSignalSemaphoreInfos = &signal_info
    }
    res := vk.QueueSubmit2(gd.transfer_queue, 1, &submit_info, 0)
    if res != .SUCCESS {
        log.errorf("Failed to submit to transfer queue: %v", res)
    }

    // CPU wait
    wait_info := vk.SemaphoreWaitInfo {
        sType = .SEMAPHORE_WAIT_INFO,
        pNext = nil,
        flags = nil,
        semaphoreCount = 1,
        pSemaphores = semaphore,
        pValues = &new_timeline_value
    }
    if vk.WaitSemaphores(gd.device, &wait_info, max(u64)) != .SUCCESS {
        log.error("Failed to wait for transfer semaphore.")
    }
    gd.transfers_completed += 1

    return handle
}

get_image :: proc(gd: ^Graphics_Device, handle: Texture_Handle) -> (^Image, bool) {
    return hm.get(&gd.images, hm.Handle(handle))
}

get_image_vkhandle :: proc(gd: ^Graphics_Device, handle: Texture_Handle) -> (h: vk.Image, ok: bool) {
    im := hm.get(&gd.images, hm.Handle(handle)) or_return
    return im.image, true
}

// Blocking function for creating a new GPU image with initial data
// Use when the data is small or if you don't care about stalling
sync_create_image_with_data :: proc(
    gd: ^Graphics_Device,
    create_info: ^Image_Create,
    bytes: []byte
) -> (out_handle: Texture_Handle, ok: bool) {
    // Create image first
    out_handle = create_image(gd, create_info)
    out_image := hm.get(&gd.images, hm.Handle(out_handle)) or_return

    extent := &create_info.extent

    cb_idx := CommandBuffer_Index(in_flight_idx(gd))
    cb := gd.transfer_command_buffers[cb_idx]
    semaphore := hm.get(&gd.semaphores, hm.Handle(gd.transfer_timeline)) or_return

    // Staging buffer
    sb := hm.get(&gd.buffers, hm.Handle(gd.staging_buffer)) or_return
    sb_ptr := sb.alloc_info.mapped_data

    if !create_info.has_mipmaps {
        create_info.mip_count = 1
    }
    
    aspect_mask : vk.ImageAspectFlags = {.COLOR} 
    if create_info.format == .D32_SFLOAT {
        aspect_mask = {.DEPTH}
    }

    bytes_per_pixel := u32(vk_format_pixel_size(create_info.format))

    bytes_size := len(bytes)
    if bytes_size > STAGING_BUFFER_SIZE {
        log.errorf("Width: %v\nHeight: %v\nDepth: %v", create_info.extent.width, create_info.extent.height, create_info.extent.depth)
        assert(false, "Image too big for staging buffer. Nick, stop being lazy.")
    }
    amount_transferred := 0

    // Upload the image data in STAGING_BUFFER_SIZE-sized chunks
    // Each iteration waits for the queue submit to complete before continuing
    for amount_transferred < bytes_size {
        remaining_bytes := bytes_size - amount_transferred
        iter_size := min(remaining_bytes, STAGING_BUFFER_SIZE)

        // Copy data to staging buffer
        p := &bytes[amount_transferred]
        mem.copy(sb_ptr, p, iter_size)

        // Begin transfer command buffer
        begin_info := vk.CommandBufferBeginInfo {
            sType = .COMMAND_BUFFER_BEGIN_INFO,
            pNext = nil,
            flags = {.ONE_TIME_SUBMIT},
            pInheritanceInfo = nil
        }
        vk.BeginCommandBuffer(cb, &begin_info)

        // Record image layout transition on first iteration
        if amount_transferred == 0 {
            barriers := []Image_Barrier {
                {
                    src_stage_mask = {},
                    src_access_mask = {},
                    dst_stage_mask = {.ALL_COMMANDS},
                    dst_access_mask = {.MEMORY_READ,.MEMORY_WRITE},
                    old_layout = .UNDEFINED,
                    new_layout = .TRANSFER_DST_OPTIMAL,
                    image = out_image.image,
                    subresource_range = {
                        aspectMask = aspect_mask,
                        baseMipLevel = 0,
                        levelCount = create_info.mip_count,
                        baseArrayLayer = 0,
                        layerCount = create_info.array_layers
                    }
                }
            }
            cmd_transfer_pipeline_barriers(gd, cb_idx, {}, barriers)
        }

        min_blocksize := u32(vk_format_block_size(create_info.format))
        buffer_offset: vk.DeviceSize = 0
        copies := make([dynamic]vk.BufferImageCopy, 0, create_info.array_layers * create_info.mip_count, context.temp_allocator)
        for current_layer in 0..<create_info.array_layers {
            for current_mip in 0..<create_info.mip_count {
                copy_extent := vk.Extent3D {
                    width = max(extent.width >> current_mip, 1),
                    height = max(extent.height >> current_mip, 1),
                    depth = max(extent.depth >> current_mip, 1),
                }

                //if copy_extent.width > 4 && copy_extent.height > 4 && copy_extent.depth > 0 {
                {
                    image_copy := vk.BufferImageCopy {
                        bufferOffset = buffer_offset,
                        bufferRowLength = 0,
                        bufferImageHeight = 0,
                        imageSubresource = {
                            aspectMask = aspect_mask,
                            mipLevel = current_mip,
                            baseArrayLayer = current_layer,
                            layerCount = 1
                        },
                        imageOffset = {
                            x = 0,
                            y = 0,
                            z = 0
                        },
                        imageExtent = copy_extent
                    }
                    append(&copies, image_copy)
                }

                buffer_offset += max(vk.DeviceSize(min_blocksize), vk.DeviceSize(bytes_per_pixel * copy_extent.width * copy_extent.height * copy_extent.depth))
            }
        }
        vk.CmdCopyBufferToImage(cb, sb.buffer, out_image.image, .TRANSFER_DST_OPTIMAL, u32(len(copies)), raw_data(copies))

        // Record queue family ownership transfer on last iteration
        // @TODO: Barrier is overly opinionated about future usage
        if remaining_bytes <= STAGING_BUFFER_SIZE {
            barriers := []Image_Barrier {
                {
                    src_stage_mask = {.TRANSFER},
                    src_access_mask = {.TRANSFER_WRITE},
                    dst_stage_mask = {.ALL_COMMANDS},                      // Ignored during release operation
                    dst_access_mask = {.MEMORY_READ},                      // Ignored during release operation
                    old_layout = .TRANSFER_DST_OPTIMAL,
                    new_layout = .SHADER_READ_ONLY_OPTIMAL,
                    src_queue_family = gd.transfer_queue_family,
                    dst_queue_family = gd.gfx_queue_family,
                    image = out_image.image,
                    subresource_range = {
                        aspectMask = {.COLOR},
                        baseMipLevel = 0,
                        levelCount = create_info.mip_count,
                        baseArrayLayer = 0,
                        layerCount = create_info.array_layers
                    }
                }
            }
            cmd_transfer_pipeline_barriers(gd, cb_idx, {}, barriers)
        }

        vk.EndCommandBuffer(cb)

        // Actually submit this lonesome command to the transfer queue
        cb_info := vk.CommandBufferSubmitInfo {
            sType = .COMMAND_BUFFER_SUBMIT_INFO,
            pNext = nil,
            commandBuffer = cb,
            deviceMask = 0
        }

        // Increment transfer timeline counter
        new_timeline_value := gd.transfers_completed + 1
        signal_info := vk.SemaphoreSubmitInfo {
            sType = .SEMAPHORE_SUBMIT_INFO,
            pNext = nil,
            semaphore = semaphore^,
            value = new_timeline_value,
            stageMask = {.ALL_COMMANDS},    // @TODO: This is a bit heavy-handed
            deviceIndex = 0
        }

        submit_info := vk.SubmitInfo2 {
            sType = .SUBMIT_INFO_2,
            pNext = nil,
            flags = nil,
            commandBufferInfoCount = 1,
            pCommandBufferInfos = &cb_info,
            signalSemaphoreInfoCount = 1,
            pSignalSemaphoreInfos = &signal_info
        }
        res := vk.QueueSubmit2(gd.transfer_queue, 1, &submit_info, 0)
        if res != .SUCCESS {
            log.errorf("Failed to submit to transfer queue: %v", res)
        }

        // CPU wait
        wait_info := vk.SemaphoreWaitInfo {
            sType = .SEMAPHORE_WAIT_INFO,
            pNext = nil,
            flags = nil,
            semaphoreCount = 1,
            pSemaphores = semaphore,
            pValues = &new_timeline_value
        }
        if vk.WaitSemaphores(gd.device, &wait_info, max(u64)) != .SUCCESS {
            log.error("Failed to wait for transfer semaphore.")
        }
        log.debugf("Transferred %v bytes of image data to image handle %v", iter_size, out_image)

        gd.transfers_completed += 1
        amount_transferred += iter_size
    }

    pending_image := Pending_Image {
        handle = out_handle,
        old_layout = .TRANSFER_DST_OPTIMAL,
        new_layout = .SHADER_READ_ONLY_OPTIMAL,
        aspect_mask = aspect_mask,
        src_queue_family = gd.transfer_queue_family,
        array_layers = create_info.array_layers,
        mip_count = create_info.mip_count,
    }
    queue.push_back(&gd.pending_images, pending_image)

    ok = true
    return
}

delete_image :: proc(gd: ^Graphics_Device, handle: Texture_Handle) -> bool {
    image := hm.get(&gd.images, hm.Handle(handle)) or_return

    image_delete := Image_Delete {
        death_frame = gd.frame_count + u64(gd.frames_in_flight),
        image = image.image,
        image_view = image.image_view,
        allocation = image.allocation
    }
    queue.append(&gd.image_deletes, image_delete)
    hm.remove(&gd.images, hm.Handle(handle))

    return true
}

acquire_swapchain_image :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    sync: ^SyncInfo,
) -> (image_idx: u32, res: vk.Result) {
    idx := in_flight_idx(gd)
    sem, sem_ok := get_semaphore(gd, gd.acquire_semaphores[idx])
    if !sem_ok {
        log.error("Couldn't get acquire semaphore.")
    }

    res = vk.AcquireNextImageKHR(gd.device, gd.swapchain, max(u64), sem^, 0, &image_idx)
    if res != .SUCCESS {
        return
    }

    // Define execution and memory dependencies surrounding swapchain image acquire

    // Wait on swapchain image acquire semaphore
    // and signal when we're done drawing on a different semaphore
    add_wait_op(gd, sync, gd.acquire_semaphores[idx])
    add_signal_op(gd, sync, gd.present_semaphores[image_idx])

    swapchain_image_handle := gd.swapchain_images[image_idx]

    // Memory barrier between swapchain acquire and rendering
    swapchain_vkimage, _ := get_image_vkhandle(gd, swapchain_image_handle)
    cmd_gfx_pipeline_barriers(gd, cb_idx, {}, {
        Image_Barrier {
            src_stage_mask = {.ALL_COMMANDS},
            src_access_mask = {.MEMORY_READ},
            dst_stage_mask = {.COLOR_ATTACHMENT_OUTPUT},
            dst_access_mask = {.MEMORY_WRITE},
            old_layout = .UNDEFINED,
            new_layout = .COLOR_ATTACHMENT_OPTIMAL,
            src_queue_family = gd.gfx_queue_family,
            dst_queue_family = gd.gfx_queue_family,
            image = swapchain_vkimage,
            subresource_range = vk.ImageSubresourceRange {
                aspectMask = {.COLOR},
                baseMipLevel = 0,
                levelCount = 1,
                baseArrayLayer = 0,
                layerCount = 1
            }
        }
    })


    return
}

present_swapchain_image :: proc(gd: ^Graphics_Device, image_idx: ^u32) -> vk.Result {
    sem, sem_ok := get_semaphore(gd, gd.present_semaphores[image_idx^])
    if !sem_ok {
        log.error("Couldn't get present semaphore.")
    }
    info := vk.PresentInfoKHR {
        sType = .PRESENT_INFO_KHR,
        pNext = nil,
        waitSemaphoreCount = 1,
        pWaitSemaphores = sem,
        swapchainCount = 1,
        pSwapchains = &gd.swapchain,
        pImageIndices = image_idx,
        pResults = nil
    }
    res := vk.QueuePresentKHR(gd.gfx_queue, &info)
    if res != .SUCCESS {
        return res
    }
    return res
}


CommandBuffer_Index :: distinct u32

begin_compute_command_buffer :: proc(
    gd: ^Graphics_Device,
    timeline_semaphore: Semaphore_Handle
) -> CommandBuffer_Index {
    // Sync point where we wait if there are already N frames in the gfx queue
    if gd.frame_count >= u64(gd.frames_in_flight) {
        // Wait on timeline semaphore before starting command buffer execution
        wait_value := gd.frame_count - u64(gd.frames_in_flight) + 1

        // CPU-sync to prevent CPU from getting further ahead than
        // the number of frames in flight
        sem, ok := get_semaphore(gd, timeline_semaphore)
        if !ok {
            log.error("Couldn't find semaphore for CPU-sync")
        }
        info := vk.SemaphoreWaitInfo {
            sType = .SEMAPHORE_WAIT_INFO,
            pNext = nil,
            flags = nil,
            semaphoreCount = 1,
            pSemaphores = sem,
            pValues = &wait_value
        }
        res := vk.WaitSemaphores(gd.device, &info, max(u64))
        if res != .SUCCESS {
            log.errorf("CPU failed to wait for timeline semaphore: %v", res)
        }
    }

    cb_idx := CommandBuffer_Index(gd.next_compute_command_buffer)
    gd.next_compute_command_buffer = (gd.next_compute_command_buffer + 1) % gd.frames_in_flight

    cb := gd.compute_command_buffers[cb_idx]
    info := vk.CommandBufferBeginInfo {
        sType = .COMMAND_BUFFER_BEGIN_INFO,
        pNext = nil,
        flags = {.ONE_TIME_SUBMIT},
        pInheritanceInfo = nil
    }
    if vk.BeginCommandBuffer(cb, &info) != .SUCCESS {
        log.error("Unable to begin compute command buffer.")
    }

    return cb_idx
}

begin_gfx_command_buffer :: proc(
    gd: ^Graphics_Device
) -> CommandBuffer_Index {

    cb_idx := CommandBuffer_Index(gd.next_gfx_command_buffer)
    gd.next_gfx_command_buffer = (gd.next_gfx_command_buffer + 1) % gd.frames_in_flight

    cb := gd.gfx_command_buffers[cb_idx]
    info := vk.CommandBufferBeginInfo {
        sType = .COMMAND_BUFFER_BEGIN_INFO,
        pNext = nil,
        flags = {.ONE_TIME_SUBMIT},
        pInheritanceInfo = nil
    }
    if vk.BeginCommandBuffer(cb, &info) != .SUCCESS {
        log.error("Unable to begin gfx command buffer.")
    }

    // Do per-frame work that has to happen for any gfx command buffer
    {
        // Process buffer delete queue
        for queue.len(gd.buffer_deletes) > 0 && queue.front_ptr(&gd.buffer_deletes).death_frame == gd.frame_count {
            buffer := queue.pop_front(&gd.buffer_deletes)
            log.debugf("Destroying buffer %v...", buffer.buffer)
            vma.destroy_buffer(gd.allocator, buffer.buffer, buffer.allocation)
        }

        // Process image delete queue
        for queue.len(gd.image_deletes) > 0 && queue.front_ptr(&gd.image_deletes).death_frame == gd.frame_count {
            image := queue.pop_front(&gd.image_deletes)
            log.debugf("Destroying image %v...", image.image)
            vk.DestroyImageView(gd.device, image.image_view, gd.alloc_callbacks)
            vma.destroy_image(gd.allocator, image.image, image.allocation)
        }

        // Process acceleration structure delete queue
        for queue.len(gd.AS_deletes) > 0 && queue.front_ptr(&gd.AS_deletes).death_frame == gd.frame_count {
            as := queue.pop_front(&gd.AS_deletes)
            log.debugf("Destroying acceleration structure %v...", as.handle)
            vk.DestroyAccelerationStructureKHR(gd.device, as.handle, gd.alloc_callbacks)
        }

        d_image_infos := make([dynamic]vk.DescriptorImageInfo, context.temp_allocator)
        d_writes := make([dynamic]vk.WriteDescriptorSet, context.temp_allocator)

        // Record queue family ownership transfer for in-flight images
        {
            write_ct := 0
            for queue.len(gd.pending_images) > 0 {
                pending_image := queue.pop_front(&gd.pending_images)
                underlying_image, ok := get_image(gd, pending_image.handle)
                if !ok {
                    log.errorf("Couldn't get pending image from handle %v", pending_image.handle)
                    continue
                }

                // @TODO: Fix the barrier masks to not suck
                barriers := []Image_Barrier {
                    {
                        src_stage_mask = {.ALL_COMMANDS},                       // Ignored in acquire operation
                        src_access_mask = {.MEMORY_READ,.MEMORY_WRITE},         // Ignored in acquire operation
                        dst_stage_mask = {.ALL_COMMANDS},
                        dst_access_mask = {.MEMORY_READ,.MEMORY_WRITE},
                        old_layout = pending_image.old_layout,
                        new_layout = pending_image.new_layout,
                        src_queue_family = pending_image.src_queue_family,
                        dst_queue_family = gd.gfx_queue_family,
                        image = underlying_image.image,
                        subresource_range = {
                            aspectMask = pending_image.aspect_mask,
                            baseMipLevel = 0,
                            levelCount = pending_image.mip_count,
                            baseArrayLayer = 0,
                            layerCount = pending_image.array_layers
                        }
                    }
                }
                cmd_gfx_pipeline_barriers(gd, cb_idx, {}, barriers)

                // Save descriptor update data
                append(&d_image_infos, vk.DescriptorImageInfo {
                    imageView = underlying_image.image_view,
                    imageLayout = .SHADER_READ_ONLY_OPTIMAL
                })
                append(&d_writes, vk.WriteDescriptorSet {
                    sType = .WRITE_DESCRIPTOR_SET,
                    pNext = nil,
                    dstSet = gd.descriptor_set,
                    dstBinding = u32(Bindless_Descriptor_Bindings.Images),
                    dstArrayElement = u32(pending_image.handle.index),
                    descriptorCount = 1,
                    descriptorType = .SAMPLED_IMAGE,
                    pImageInfo = &d_image_infos[write_ct]
                })
                write_ct += 1
            }
        }

        // Update sampled image descriptors
        if len(d_writes) > 0 {
            vk.UpdateDescriptorSets(gd.device, u32(len(d_writes)), &d_writes[0], 0, nil)
        }
    }

    return cb_idx
}

build_submit_infos :: proc(
    gd: ^Graphics_Device,
    submit_infos: ^[dynamic]vk.SemaphoreSubmitInfoKHR,
    semaphore_ops: [dynamic]Semaphore_Op
) -> bool {
    count := len(semaphore_ops)
    for i := 0; i < count; i += 1 {
        sem := hm.get(&gd.semaphores, hm.Handle(semaphore_ops[i].semaphore)) or_return
        submit_infos[i] = vk.SemaphoreSubmitInfo{
            sType = .SEMAPHORE_SUBMIT_INFO_KHR,
            pNext = nil,
            semaphore = sem^,
            value = semaphore_ops[i].value,
            stageMask = {.ALL_COMMANDS},    // @TODO: Is this heavy-handed or appropriate?
            deviceIndex = 0
        }
    }
    return true
}

submit_compute_command_buffer :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    sync: ^SyncInfo
) {
    cb := gd.compute_command_buffers[cb_idx]
    if vk.EndCommandBuffer(cb) != .SUCCESS {
        log.error("Unable to end compute command buffer")
    }

    cb_info := vk.CommandBufferSubmitInfo{
        sType = .COMMAND_BUFFER_SUBMIT_INFO_KHR,
        pNext = nil,
        commandBuffer = cb,
        deviceMask = 0
    }

    wait_infos := make([dynamic]vk.SemaphoreSubmitInfoKHR, len(sync.wait_ops), allocator = context.temp_allocator)
    signal_infos := make([dynamic]vk.SemaphoreSubmitInfoKHR, len(sync.signal_ops), allocator = context.temp_allocator)
    build_submit_infos(gd, &wait_infos, sync.wait_ops)
    build_submit_infos(gd, &signal_infos, sync.signal_ops)

    info := vk.SubmitInfo2 {
        sType = .SUBMIT_INFO_2,
        pNext = nil,
        flags = nil,
        waitSemaphoreInfoCount = u32(len(wait_infos)),
        pWaitSemaphoreInfos = raw_data(wait_infos),
        signalSemaphoreInfoCount = u32(len(signal_infos)),
        pSignalSemaphoreInfos = raw_data(signal_infos),
        commandBufferInfoCount = 1,
        pCommandBufferInfos = &cb_info
    }
    res := vk.QueueSubmit2(gd.compute_queue, 1, &info, 0)
    if res != .SUCCESS {
        log.errorf("Unable to submit compute command buffer: %v", res)
    }
}

submit_gfx_command_buffer :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    sync: ^SyncInfo
) {
    cb := gd.gfx_command_buffers[cb_idx]
    if vk.EndCommandBuffer(cb) != .SUCCESS {
        log.error("Unable to end gfx command buffer")
    }

    cb_info := vk.CommandBufferSubmitInfo{
        sType = .COMMAND_BUFFER_SUBMIT_INFO_KHR,
        pNext = nil,
        commandBuffer = cb,
        deviceMask = 0
    }

    // Make semaphore submit infos
    wait_infos := make([dynamic]vk.SemaphoreSubmitInfoKHR, len(sync.wait_ops), allocator = context.temp_allocator)
    signal_infos := make([dynamic]vk.SemaphoreSubmitInfoKHR, len(sync.signal_ops), allocator = context.temp_allocator)
    build_submit_infos(gd, &wait_infos, sync.wait_ops)
    build_submit_infos(gd, &signal_infos, sync.signal_ops)

    info := vk.SubmitInfo2{
        sType = .SUBMIT_INFO_2_KHR,
        pNext = nil,
        flags = nil,
        waitSemaphoreInfoCount = u32(len(wait_infos)),
        pWaitSemaphoreInfos = raw_data(wait_infos),
        signalSemaphoreInfoCount = u32(len(signal_infos)),
        pSignalSemaphoreInfos = raw_data(signal_infos),
        commandBufferInfoCount = 1,
        pCommandBufferInfos = &cb_info
    }
    if vk.QueueSubmit2(gd.gfx_queue, 1, &info, 0) != .SUCCESS {
        log.error("Unable to submit gfx command buffer")
    }
}

submit_gfx_and_present :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    sync: ^SyncInfo,
    swapchain_idx: ^u32
) -> vk.Result {
    // Memory barrier between rendering to swapchain image and swapchain present
    swapchain_image, _ := get_image_vkhandle(gd, gd.swapchain_images[swapchain_idx^])
    cmd_gfx_pipeline_barriers(gd, cb_idx, {}, {
        Image_Barrier {
            src_stage_mask = {.COLOR_ATTACHMENT_OUTPUT},
            src_access_mask = {.MEMORY_WRITE},
            dst_stage_mask = {.ALL_COMMANDS},
            dst_access_mask = {.MEMORY_READ},
            old_layout = .COLOR_ATTACHMENT_OPTIMAL,
            new_layout = .PRESENT_SRC_KHR,
            src_queue_family = gd.gfx_queue_family,
            dst_queue_family = gd.gfx_queue_family,
            image = swapchain_image,
            subresource_range = vk.ImageSubresourceRange {
                aspectMask = {.COLOR},
                baseMipLevel = 0,
                levelCount = 1,
                baseArrayLayer = 0,
                layerCount = 1
            }
        }
    })

    submit_gfx_command_buffer(gd, cb_idx, sync)
    present_res := present_swapchain_image(gd, swapchain_idx)

    gd.frame_count += 1

    return present_res
}











Framebuffer :: struct {
    color_images: [8]Texture_Handle,
    depth_image: Texture_Handle,
    resolution: hlsl.uint2,
    clear_color: hlsl.float4,
    color_load_op: vk.AttachmentLoadOp,
    depth_load_op: vk.AttachmentLoadOp
}

cmd_begin_render_pass :: proc(gd: ^Graphics_Device, cb_idx: CommandBuffer_Index, framebuffer: ^Framebuffer) {
    cb := gd.gfx_command_buffers[cb_idx]

    iv, ok := hm.get(&gd.images, hm.Handle(framebuffer.color_images[0]))
    color_attachment := vk.RenderingAttachmentInfo {
        sType = .RENDERING_ATTACHMENT_INFO_KHR,
        pNext = nil,
        imageView = iv.image_view,
        imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
        loadOp = framebuffer.color_load_op,
        storeOp = .STORE,
        clearValue = vk.ClearValue {
            color = vk.ClearColorValue {
                float32 = cast([4]f32)framebuffer.clear_color
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

    depth_attachment: vk.RenderingAttachmentInfo
    depth_image, ok2 := hm.get(&gd.images, hm.Handle(framebuffer.depth_image))
    if ok2 {
        depth_attachment = vk.RenderingAttachmentInfo {
            sType = .RENDERING_ATTACHMENT_INFO,
            pNext = nil,
            imageView = depth_image.image_view,
            imageLayout = .DEPTH_ATTACHMENT_OPTIMAL,
            loadOp = framebuffer.depth_load_op,
            storeOp = .STORE,
            clearValue = {
                depthStencil = {
                    depth = 0.0
                }
            }
        }
        info.pDepthAttachment = &depth_attachment
    }


    vk.CmdBeginRendering(cb, &info)
}

cmd_bind_gfx_descriptor_set :: proc(gd: ^Graphics_Device, cb_idx: CommandBuffer_Index) {
    cb := gd.gfx_command_buffers[cb_idx]
    vk.CmdBindDescriptorSets(cb, .GRAPHICS, gd.gfx_pipeline_layout, 0, 1, &gd.descriptor_set, 0, nil)
}

cmd_bind_gfx_pipeline :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    handle: Pipeline_Handle
) -> bool {
    cb := gd.gfx_command_buffers[cb_idx]
    pipeline := hm.get(&gd.pipelines, hm.Handle(handle)) or_return
    vk.CmdBindPipeline(cb, .GRAPHICS, pipeline^)
    return true
}

cmd_bind_compute_pipeline :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    handle: Pipeline_Handle
) -> bool {
    cb := gd.compute_command_buffers[cb_idx]
    pipeline := hm.get(&gd.pipelines, hm.Handle(handle)) or_return
    vk.CmdBindPipeline(cb, .COMPUTE, pipeline^)
    return true
}

cmd_draw :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    vtx_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32
) {
    cb := gd.gfx_command_buffers[cb_idx]
    vk.CmdDraw(cb, vtx_count, instance_count, first_vertex, first_instance)
}

cmd_draw_indexed :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32
) {
    cb := gd.gfx_command_buffers[cb_idx]
    vk.CmdDrawIndexed(cb, index_count, instance_count, first_index, vertex_offset, first_instance)
}

cmd_dispatch :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    group_countx: u32,
    group_county: u32,
    group_countz: u32,
) {
    cb := gd.compute_command_buffers[cb_idx]
    vk.CmdDispatch(cb, group_countx, group_county, group_countz)
}

Viewport :: vk.Viewport
cmd_set_viewport :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    first_viewport: u32,
    viewports: []Viewport
) {
    cb := gd.gfx_command_buffers[cb_idx]
    vk.CmdSetViewport(cb, first_viewport, u32(len(viewports)), raw_data(viewports))
}

Scissor :: vk.Rect2D
cmd_set_scissor :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    first_scissor: u32,
    scissors: []Scissor
) {
    cb := gd.gfx_command_buffers[cb_idx]
    vk.CmdSetScissor(cb, first_scissor, u32(len(scissors)), raw_data(scissors))
}

cmd_push_constants_gfx :: proc(gd: ^Graphics_Device, cb_idx: CommandBuffer_Index, in_struct: ^$Struct_Type) {
    cb := gd.gfx_command_buffers[cb_idx]
    byte_count : u32 = u32(size_of(Struct_Type))
    vk.CmdPushConstants(cb, gd.gfx_pipeline_layout, {.VERTEX,.FRAGMENT}, 0, byte_count, in_struct)
}

cmd_push_constants_compute :: proc(gd: ^Graphics_Device, cb_idx: CommandBuffer_Index, in_struct: ^$Struct_Type) {
    cb := gd.compute_command_buffers[cb_idx]
    byte_count : u32 = u32(size_of(Struct_Type))
    vk.CmdPushConstants(cb, gd.compute_pipeline_layout, {.COMPUTE}, 0, byte_count, in_struct)
}

cmd_bind_index_buffer :: proc(gd: ^Graphics_Device, cb_idx: CommandBuffer_Index, buffer: Buffer_Handle) -> bool {
    cb := gd.gfx_command_buffers[cb_idx]
    b := get_buffer(gd, buffer) or_return
    vk.CmdBindIndexBuffer(cb, b.buffer, 0, .UINT16)

    return true
}

cmd_draw_indexed_indirect :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    draw_buffer_handle: Buffer_Handle,
    offset: u64,
    draw_count: u32
) -> bool {
    cb := gd.gfx_command_buffers[cb_idx]
    draw_buffer := get_buffer(gd, draw_buffer_handle) or_return
    vk.CmdDrawIndexedIndirect(
        cb,
        draw_buffer.buffer,
        vk.DeviceSize(offset),
        draw_count,
        size_of(vk.DrawIndexedIndirectCommand)
    )

    return true
}

cmd_end_render_pass :: proc(gd: ^Graphics_Device, cb_idx: CommandBuffer_Index) {
    cb := gd.gfx_command_buffers[cb_idx]
    vk.CmdEndRendering(cb)
}

Semaphore_Info :: struct {
    type: vk.SemaphoreType,
    init_value: u64,
    name: cstring
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

    if len(info.name) > 0 {
        assign_debug_name(gd.device, .SEMAPHORE, u64(s), info.name)
    } else {
        log.warn("Semaphore was created without a debug name")
    }


    return Semaphore_Handle(hm.insert(&gd.semaphores, s))
}

check_timeline_semaphore :: proc(gd: ^Graphics_Device, handle: Semaphore_Handle) -> (val: u64, ok: bool) {
    sem := hm.get(&gd.semaphores, hm.Handle(handle)) or_return
    v: u64
    if vk.GetSemaphoreCounterValue(gd.device, sem^, &v) != .SUCCESS {
        log.error("Failed to read value from timeline semaphore")
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

// Defines a memory dependency
// on one buffer range
Buffer_Barrier :: struct {
    src_stage_mask: vk.PipelineStageFlags2,
    src_access_mask: vk.AccessFlags2,
    dst_stage_mask: vk.PipelineStageFlags2,
    dst_access_mask: vk.AccessFlags2,
    src_queue_family: u32,
    dst_queue_family: u32,
    buffer: vk.Buffer,
    offset: vk.DeviceSize,
    size: vk.DeviceSize,
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
    image: vk.Image,
    subresource_range: vk.ImageSubresourceRange
}

// Inserts an arbitrary number of memory barriers
// into the gfx command buffer at this point
cmd_gfx_pipeline_barriers :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    buffer_barriers: []Buffer_Barrier,
    image_barriers: []Image_Barrier
) {
    cb := gd.gfx_command_buffers[cb_idx]

    buf_barriers := make([dynamic]vk.BufferMemoryBarrier2, 0, len(buffer_barriers), allocator = context.temp_allocator)
    for barrier in buffer_barriers {
        append(&buf_barriers, vk.BufferMemoryBarrier2 {
            sType = .BUFFER_MEMORY_BARRIER_2,
            pNext = nil,
            srcStageMask = barrier.src_stage_mask,
            srcAccessMask = barrier.src_access_mask,
            dstStageMask = barrier.dst_stage_mask,
            dstAccessMask = barrier.dst_access_mask,
            srcQueueFamilyIndex = barrier.src_queue_family,
            dstQueueFamilyIndex = barrier.dst_queue_family,
            buffer = barrier.buffer,
            offset = barrier.offset,
            size = barrier.size,
        })
    }

    im_barriers := make([dynamic]vk.ImageMemoryBarrier2, 0, len(image_barriers), context.temp_allocator)
    for barrier in image_barriers {
        using barrier

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
                image = image,
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
        bufferMemoryBarrierCount = u32(len(buf_barriers)),
        pBufferMemoryBarriers = raw_data(buf_barriers),
        imageMemoryBarrierCount = u32(len(im_barriers)),
        pImageMemoryBarriers = raw_data(im_barriers)
    }
    vk.CmdPipelineBarrier2(cb, &info)
}

// Inserts an arbitrary number of memory barriers
// into the compute command buffer at this point
cmd_compute_pipeline_barriers :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    buffer_barriers: []Buffer_Barrier,
    image_barriers: []Image_Barrier
) {
    cb := gd.compute_command_buffers[cb_idx]

    buf_barriers := make([dynamic]vk.BufferMemoryBarrier2, 0, len(buffer_barriers), allocator = context.temp_allocator)
    for barrier in buffer_barriers {
        append(&buf_barriers, vk.BufferMemoryBarrier2 {
            sType = .BUFFER_MEMORY_BARRIER_2,
            pNext = nil,
            srcStageMask = barrier.src_stage_mask,
            srcAccessMask = barrier.src_access_mask,
            dstStageMask = barrier.dst_stage_mask,
            dstAccessMask = barrier.dst_access_mask,
            srcQueueFamilyIndex = barrier.src_queue_family,
            dstQueueFamilyIndex = barrier.dst_queue_family,
            buffer = barrier.buffer,
            offset = barrier.offset,
            size = barrier.size,
        })
    }

    im_barriers := make([dynamic]vk.ImageMemoryBarrier2, 0, len(buffer_barriers), allocator = context.temp_allocator)
    for barrier in image_barriers {
        using barrier

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
                image = image,
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
        bufferMemoryBarrierCount = u32(len(buf_barriers)),
        pBufferMemoryBarriers = raw_data(buf_barriers),
        imageMemoryBarrierCount = u32(len(im_barriers)),
        pImageMemoryBarriers = raw_data(im_barriers)
    }
    vk.CmdPipelineBarrier2(cb, &info)
}

// Inserts an arbitrary number of memory barriers
// into the transfer command buffer at this point
cmd_transfer_pipeline_barriers :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    buffer_barriers: []Buffer_Barrier,
    image_barriers: []Image_Barrier
) {
    cb := gd.transfer_command_buffers[cb_idx]

    buf_barriers := make([dynamic]vk.BufferMemoryBarrier2, 0, len(buffer_barriers), allocator = context.temp_allocator)
    for barrier in buffer_barriers {
        append(&buf_barriers, vk.BufferMemoryBarrier2 {
            sType = .BUFFER_MEMORY_BARRIER_2,
            pNext = nil,
            srcStageMask = barrier.src_stage_mask,
            srcAccessMask = barrier.src_access_mask,
            dstStageMask = barrier.dst_stage_mask,
            dstAccessMask = barrier.dst_access_mask,
            srcQueueFamilyIndex = barrier.src_queue_family,
            dstQueueFamilyIndex = barrier.dst_queue_family,
            buffer = barrier.buffer,
            offset = barrier.offset,
            size = barrier.size,
        })
    }

    im_barriers := make([dynamic]vk.ImageMemoryBarrier2, 0, len(image_barriers), context.temp_allocator)
    for barrier in image_barriers {
        using barrier

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
                image = image,
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
        bufferMemoryBarrierCount = u32(len(buf_barriers)),
        pBufferMemoryBarriers = raw_data(buf_barriers),
        imageMemoryBarrierCount = u32(len(im_barriers)),
        pImageMemoryBarriers = raw_data(im_barriers)
    }
    vk.CmdPipelineBarrier2(cb, &info)
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
    sample_mask: ^vk.SampleMask,
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
    blend_constants: hlsl.float4,
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

create_shader_module :: proc(gd: ^Graphics_Device, code: []u32) -> vk.ShaderModule {
    info := vk.ShaderModuleCreateInfo {
        sType = .SHADER_MODULE_CREATE_INFO,
        pNext = nil,
        flags = nil,
        codeSize = size_of(u32) * len(code),
        pCode = raw_data(code)
    }
    mod: vk.ShaderModule
    p_info := &info
    if vk.CreateShaderModule(gd.device, p_info, gd.alloc_callbacks, &mod) != .SUCCESS {
        log.error("Failed to create shader module.")
    }
    return mod
}

Pipeline_Handle :: distinct hm.Handle
GraphicsPipelineInfo :: struct {
    vertex_shader_bytecode: []u32,
    fragment_shader_bytecode: []u32,
    vertex_spec_constants: []i32,
    fragment_spec_constants: []i32,
    input_assembly_state: Input_Assembly_State,
    tessellation_state: Tessellation_State,
    rasterization_state: Rasterization_State,
    multisample_state: Multisample_State,
    depthstencil_state: DepthStencil_State,
    colorblend_state: ColorBlend_State,
    renderpass_state: PipelineRenderpass_Info,
    name: string,
}

create_graphics_pipelines :: proc(gd: ^Graphics_Device, infos: []GraphicsPipelineInfo) -> [dynamic]Pipeline_Handle {
    pipeline_count := len(infos)

    // Output dynamic array of pipeline handles
    handles := make([dynamic]Pipeline_Handle, pipeline_count, context.temp_allocator)


    // One dynamic array for each thing in Graphics_Pipeline_Info
    pipelines := make([dynamic]vk.Pipeline, pipeline_count, context.temp_allocator)
    create_infos := make([dynamic]vk.GraphicsPipelineCreateInfo, pipeline_count, context.temp_allocator)

    shader_modules := make([dynamic]vk.ShaderModule, 2 * pipeline_count, context.temp_allocator)
    defer for module in shader_modules {
        vk.DestroyShaderModule(gd.device, module, gd.alloc_callbacks)
    }    

    shader_infos := make([dynamic]vk.PipelineShaderStageCreateInfo, 2 * pipeline_count, context.temp_allocator)
    input_assembly_states := make([dynamic]vk.PipelineInputAssemblyStateCreateInfo, pipeline_count, context.temp_allocator)
    tessellation_states := make([dynamic]vk.PipelineTessellationStateCreateInfo, pipeline_count, context.temp_allocator)
    rasterization_states := make([dynamic]vk.PipelineRasterizationStateCreateInfo, pipeline_count, context.temp_allocator)
    multisample_states := make([dynamic]vk.PipelineMultisampleStateCreateInfo, pipeline_count, context.temp_allocator)
    sample_masks := make([dynamic]^vk.SampleMask, pipeline_count, context.temp_allocator)
    depthstencil_states := make([dynamic]vk.PipelineDepthStencilStateCreateInfo, pipeline_count, context.temp_allocator)
    colorblend_attachments := make([dynamic]vk.PipelineColorBlendAttachmentState, pipeline_count, context.temp_allocator)
    colorblend_states := make([dynamic]vk.PipelineColorBlendStateCreateInfo, pipeline_count, context.temp_allocator)
    renderpass_states := make([dynamic]vk.PipelineRenderingCreateInfo, pipeline_count, context.temp_allocator)
    spec_map_entries := make([dynamic]vk.SpecializationMapEntry, context.temp_allocator)
    spec_infos := make([dynamic]vk.SpecializationInfo, context.temp_allocator)

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
    spec_constant_offset := 0
    for info, i in infos {
        // Shader state
        log.debugf("Create vertex shader module for %v", info.name)
        shader_modules[2 * i]     = create_shader_module(gd, info.vertex_shader_bytecode)
        log.debugf("Create fragment shader module for %v", info.name)
        shader_modules[2 * i + 1] = create_shader_module(gd, info.fragment_shader_bytecode)

        vertex_spec_ptr : ^vk.SpecializationInfo = nil
        if len(info.vertex_spec_constants) > 0 {
            for constant, j in info.vertex_spec_constants {
                append(&spec_map_entries, vk.SpecializationMapEntry {
                    constantID = u32(j),
                    offset = u32(size_of(i32) * j),
                    size = size_of(i32)
                })
            }
            vertex_spec_info := vk.SpecializationInfo {
                mapEntryCount = u32(len(info.vertex_spec_constants)),
                pMapEntries = &spec_map_entries[spec_constant_offset],
                dataSize = size_of(i32) * len(info.vertex_spec_constants),
                pData = raw_data(info.vertex_spec_constants)
            }
            append(&spec_infos, vertex_spec_info)
            spec_constant_offset += len(info.vertex_spec_constants)
            vertex_spec_ptr = &spec_infos[len(spec_infos) - 1]
        }
        shader_infos[2 * i] = vk.PipelineShaderStageCreateInfo {
            sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            stage = {.VERTEX},
            module = shader_modules[2 * i],
            pName = "main",
            pSpecializationInfo = vertex_spec_ptr
        }

        fragment_spec_ptr : ^vk.SpecializationInfo = nil
        if len(info.fragment_spec_constants) > 0 {
            for constant, j in info.fragment_spec_constants {
                append(&spec_map_entries, vk.SpecializationMapEntry {
                    constantID = u32(j),
                    offset = u32(size_of(i32) * j),
                    size = size_of(i32)
                })
            }
            fragment_spec_info := vk.SpecializationInfo {
                mapEntryCount = u32(len(info.fragment_spec_constants)),
                pMapEntries = &spec_map_entries[spec_constant_offset],
                dataSize = size_of(i32) * len(info.fragment_spec_constants),
                pData = raw_data(info.fragment_spec_constants)
            }
            append(&spec_infos, fragment_spec_info)
            spec_constant_offset += len(info.fragment_spec_constants)
            fragment_spec_ptr = &spec_infos[len(spec_infos) - 1]
        }
        shader_infos[2 * i + 1] = vk.PipelineShaderStageCreateInfo {
            sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            stage = {.FRAGMENT},
            module = shader_modules[2 * i + 1],
            pName = "main",
            pSpecializationInfo = fragment_spec_ptr
        }

        // Input assembly state
        input_assembly_states[i] = vk.PipelineInputAssemblyStateCreateInfo {
            sType = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            topology = info.input_assembly_state.topology,
            primitiveRestartEnable = b32(info.input_assembly_state.primitive_restart_enabled)
        }

        // Tessellation state
        tessellation_states[i] = vk.PipelineTessellationStateCreateInfo {
            sType = .PIPELINE_TESSELLATION_STATE_CREATE_INFO,
            pNext = nil,
            patchControlPoints = info.tessellation_state.patch_control_points
        }

        // Rasterization state
        rasterization_states[i] = vk.PipelineRasterizationStateCreateInfo {
            sType = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            depthClampEnable = b32(info.rasterization_state.do_depth_clamp),
            rasterizerDiscardEnable = b32(info.rasterization_state.do_rasterizer_discard),
            polygonMode = info.rasterization_state.polygon_mode,
            cullMode = info.rasterization_state.cull_mode,
            frontFace = info.rasterization_state.front_face,
            depthBiasEnable = b32(info.rasterization_state.do_depth_bias),
            depthBiasConstantFactor = info.rasterization_state.depth_bias_constant_factor,
            depthBiasClamp = info.rasterization_state.depth_bias_clamp,
            depthBiasSlopeFactor = info.rasterization_state.depth_bias_slope_factor,
            lineWidth = info.rasterization_state.line_width
        }

        // Multisample state
        sample_masks[i] = info.multisample_state.sample_mask
        multisample_states[i] = vk.PipelineMultisampleStateCreateInfo {
            sType = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            rasterizationSamples = info.multisample_state.sample_count,
            sampleShadingEnable = b32(info.multisample_state.do_sample_shading),
            minSampleShading = info.multisample_state.min_sample_shading,
            pSampleMask = sample_masks[i]
        }

        // Depth-stencil state
        depthstencil_states[i] = vk.PipelineDepthStencilStateCreateInfo {
            sType = .PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            pNext = nil,
            flags = info.depthstencil_state.flags,
            depthTestEnable = b32(info.depthstencil_state.do_depth_test),
            depthWriteEnable = b32(info.depthstencil_state.do_depth_write),
            depthCompareOp = info.depthstencil_state.depth_compare_op,
            depthBoundsTestEnable = b32(info.depthstencil_state.do_depth_bounds_test),
            stencilTestEnable = b32(info.depthstencil_state.do_stencil_test),
            front = info.depthstencil_state.front,
            back = info.depthstencil_state.back,
            minDepthBounds = info.depthstencil_state.min_depth_bounds,
            maxDepthBounds = info.depthstencil_state.max_depth_bounds
        }

        // Color blend state
        colorblend_attachments[i] = vk.PipelineColorBlendAttachmentState {
            blendEnable = b32(info.colorblend_state.attachment.do_blend),
            srcColorBlendFactor = info.colorblend_state.attachment.src_color_blend_factor,
            dstColorBlendFactor = info.colorblend_state.attachment.dst_color_blend_factor,
            colorBlendOp = info.colorblend_state.attachment.color_blend_op,
            srcAlphaBlendFactor = info.colorblend_state.attachment.src_alpha_blend_factor,
            dstAlphaBlendFactor = info.colorblend_state.attachment.dst_alpha_blend_factor,
            alphaBlendOp = info.colorblend_state.attachment.alpha_blend_op,
            colorWriteMask = info.colorblend_state.attachment.color_write_mask
        }
        colorblend_states[i] = vk.PipelineColorBlendStateCreateInfo {
            sType = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            pNext = nil,
            flags = info.colorblend_state.flags,
            logicOpEnable = b32(info.colorblend_state.do_logic_op),
            attachmentCount = 1,
            pAttachments = &colorblend_attachments[i],
            blendConstants = cast([4]f32)info.colorblend_state.blend_constants
        }

        // Render pass state
        renderpass_states[i] = vk.PipelineRenderingCreateInfo {
            sType = .PIPELINE_RENDERING_CREATE_INFO,
            pNext = nil,
            viewMask = 0,
            colorAttachmentCount = u32(len(info.renderpass_state.color_attachment_formats)),
            pColorAttachmentFormats = raw_data(info.renderpass_state.color_attachment_formats),
            depthAttachmentFormat = info.renderpass_state.depth_attachment_format,
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
            layout = gd.gfx_pipeline_layout,
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
        &create_infos[0],
        gd.alloc_callbacks,
        &pipelines[0]
    )
    if res != .SUCCESS {
        log.error("Failed to compile graphics pipelines")
    }

    // Put newly created pipelines in the Handle_Map
    for p, i in pipelines {
        if len(infos[i].name) == 0 {
            log.warn("Creating graphics pipeline without debug name")
        } else {
            s := strings.clone_to_cstring(infos[i].name, context.temp_allocator)
            assign_debug_name(gd.device, .PIPELINE, u64(p), s)
        }
        handles[i] = Pipeline_Handle(hm.insert(&gd.pipelines, p))
    }

    return handles
}

ComputePipelineInfo :: struct {
    compute_shader_bytecode: []u32,
    name: string
}

create_compute_pipelines :: proc(gd: ^Graphics_Device, infos: []ComputePipelineInfo) -> [dynamic]Pipeline_Handle {
    info_count := u32(len(infos))
    pipeline_create_infos := make([dynamic]vk.ComputePipelineCreateInfo, 0, info_count, allocator = context.temp_allocator)
    pipelines := make([dynamic]vk.Pipeline, 1, info_count, allocator = context.temp_allocator)
    pipeline_handles := make([dynamic]Pipeline_Handle, info_count, allocator = context.temp_allocator)

    // Build list of create infos
    for info in infos {
        module := create_shader_module(gd, info.compute_shader_bytecode)
        stage := vk.PipelineShaderStageCreateInfo {
            sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            stage = {.COMPUTE},
            module = module,
            pName = "main",
            pSpecializationInfo = nil
        }
        create_info := vk.ComputePipelineCreateInfo {
            sType = .COMPUTE_PIPELINE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            stage = stage,
            layout = gd.compute_pipeline_layout,
            basePipelineHandle = 0,
            basePipelineIndex = 0,
        }
        append(&pipeline_create_infos, create_info)
    }
    vk.CreateComputePipelines(gd.device, gd.pipeline_cache, info_count, &pipeline_create_infos[0], gd.alloc_callbacks, &pipelines[0])

    // Insert pipelines into handlemap
    for pipeline, i in pipelines {
        if len(infos[i].name) == 0 {
            log.warn("Creating compute pipeline without debug name")
        } else {
            s := strings.clone_to_cstring(infos[i].name, context.temp_allocator)
            assign_debug_name(gd.device, .PIPELINE, u64(pipeline), s)
        }
        pipeline_handles[i] = Pipeline_Handle(hm.insert(&gd.pipelines, pipeline))
    }

    // Destroy lingering shader modules
    for info in pipeline_create_infos {
        vk.DestroyShaderModule(gd.device, info.stage.module, gd.alloc_callbacks)
    }

    return pipeline_handles
}



// Acceleration structure section
AS_BUFFER_ALIGNMENT :: 256
AccelerationStructure :: struct {
    handle: vk.AccelerationStructureKHR,
    offset: u32,
}
AccelerationStructureCreateInfo :: struct {
    flags: vk.AccelerationStructureCreateFlagsKHR,
    type: vk.AccelerationStructureTypeKHR
}

ASTrianglesData :: struct {
    vertex_format:  vk.Format,
	vertex_data:    vk.DeviceOrHostAddressConstKHR,
	vertex_stride:  vk.DeviceSize,
	max_vertex:     u32,
	index_type:     vk.IndexType,
	index_data:     vk.DeviceOrHostAddressConstKHR,
	transform_data: vk.DeviceOrHostAddressConstKHR,
}
ASInstancesData :: struct {
	array_of_pointers: bool,
	data:            [dynamic]vk.AccelerationStructureInstanceKHR,
}
AccelerationStructureGeometryData :: union {
    ASTrianglesData,
    ASInstancesData,
}
AccelerationStructureGeometry :: struct {
    type: vk.GeometryTypeKHR,
    geometry: AccelerationStructureGeometryData,
    flags: vk.GeometryFlagsKHR
}
AccelerationStructureBuildInfo :: struct {
    type: vk.AccelerationStructureTypeKHR,
    flags: vk.BuildAccelerationStructureFlagsKHR,
    mode: vk.BuildAccelerationStructureModeKHR,
    src: Acceleration_Structure_Handle,
    dst: vk.AccelerationStructureKHR,
    geometries: [dynamic]AccelerationStructureGeometry,
    prim_counts: []u32,
    //scratch_data: vk.DeviceOrHostAddressKHR,
    range_info: vk.AccelerationStructureBuildRangeInfoKHR,
    build_scratch_size: vk.DeviceSize,
}

AS_Delete :: struct {
    death_frame: u64,
    handle: vk.AccelerationStructureKHR,
}

make_geo_data_structs :: proc(gd: ^Graphics_Device, geometries: []AccelerationStructureGeometry) -> [dynamic]vk.AccelerationStructureGeometryKHR {
    geos := make([dynamic]vk.AccelerationStructureGeometryKHR, 0, len(geometries), context.temp_allocator)
    for geo in geometries {
        geo_data: vk.AccelerationStructureGeometryDataKHR
        switch d in geo.geometry {
            case ASTrianglesData: {
                geo_data.triangles = vk.AccelerationStructureGeometryTrianglesDataKHR {
                    sType = .ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                    pNext = nil,
                    vertexFormat = d.vertex_format,
                    vertexData = d.vertex_data,
                    vertexStride = d.vertex_stride,
                    maxVertex = d.max_vertex,
                    indexType = d.index_type,
                    indexData = d.index_data,
                    transformData = d.transform_data
                }
            }
            case ASInstancesData: {
                // Copy instance to GPU
                sync_write_buffer(gd, gd.TLAS_instance_buffer, d.data[:])
                b, _ := get_buffer(gd, gd.TLAS_instance_buffer)

                geo_data.instances = vk.AccelerationStructureGeometryInstancesDataKHR {
                    sType = .ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                    pNext = nil,
                    arrayOfPointers = b32(d.array_of_pointers),
                    data = {
                        deviceAddress = b.address
                    }
                }
            }
            case: {
                log.error("This should be unreachable")
            }
        }
        append(&geos, vk.AccelerationStructureGeometryKHR {
            sType = .ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            pNext = nil,
            geometryType = geo.type,
            geometry = geo_data,
            flags = geo.flags
        })
    }
    return geos
}

get_acceleration_structure_build_sizes :: proc(
    gd: ^Graphics_Device,
    info: AccelerationStructureBuildInfo
) -> vk.AccelerationStructureBuildSizesInfoKHR {
    size_info: vk.AccelerationStructureBuildSizesInfoKHR
    size_info.sType = .ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR

    geos := make_geo_data_structs(gd, info.geometries[:])

    build_info := vk.AccelerationStructureBuildGeometryInfoKHR {
        sType = .ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        pNext = nil,
        type = info.type,
        flags = info.flags,
        mode = info.mode,
        // srcAccelerationStructure = info.src,
        // dstAccelerationStructure = info.dst,
        geometryCount = u32(len(geos)),
        pGeometries = raw_data(geos),
        ppGeometries = nil,
        //scratchData = info.scratch_data
    }
    // @NOTE: A little birdy told me to never use host AS building
    vk.GetAccelerationStructureBuildSizesKHR(gd.device, .DEVICE, &build_info, raw_data(info.prim_counts), &size_info)

    return size_info
}

create_acceleration_structure :: proc(
    gd: ^Graphics_Device,
    create_info: AccelerationStructureCreateInfo,
    build_info: ^AccelerationStructureBuildInfo
) -> Acceleration_Structure_Handle {
    assert(.Raytracing in gd.support_flags)
    build_sizes := get_acceleration_structure_build_sizes(gd, build_info^)

    // Record new required scratch buffer size
    gd.AS_required_scratch_size += build_sizes.buildScratchSize

    as_buffer, _ := get_buffer(gd, gd.AS_buffer)

    src_AS, have_src := get_acceleration_structure(gd, build_info.src)
    ret_handle: Acceleration_Structure_Handle
    if have_src && src_AS.handle != 0 {
        // AS already exists and we're updating it

        build_info.mode = .UPDATE
        build_info.dst = src_AS.handle
        ret_handle = build_info.src
    } else {
        // AS does not exist yet
        
        offset := gd.BLAS_head
        if (create_info.type == .TOP_LEVEL) {
            // Will the TLAS fit into 1/nth of the remaining AS_buffer space?
            available_bytes := (AS_BUFFER_SIZE - gd.BLAS_head) / vk.DeviceSize(gd.frames_in_flight)
            if build_sizes.accelerationStructureSize > available_bytes {
                log.error("Not enough space in AS_buffer for TLAS!")
            }

            tlas_idx := u32(gd.frame_count) % gd.frames_in_flight
            offset += available_bytes * vk.DeviceSize(tlas_idx / gd.frames_in_flight)
            log.debugf("tlas_idx: %v\noffset: %v\n\n", tlas_idx, offset)
        }

        info := vk.AccelerationStructureCreateInfoKHR {
            sType = .ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            pNext = nil,
            createFlags = create_info.flags,
            buffer = as_buffer.buffer,
            offset = offset,
            size = build_sizes.accelerationStructureSize,
            type = create_info.type,
            //deviceAddress = info.device_address           // Only relevant when using the (useless) capture/replay feature
        }
        res := vk.CreateAccelerationStructureKHR(gd.device, &info, gd.alloc_callbacks, &build_info.dst)
        if res != .SUCCESS {
            log.errorf("Failed to create acceleration structure: %v", res)
        }

        if create_info.type == .BOTTOM_LEVEL {
            aligned_size := size_to_alignment(build_sizes.accelerationStructureSize, AS_BUFFER_ALIGNMENT)
            gd.BLAS_head += aligned_size
        }

        ret_handle = Acceleration_Structure_Handle(hm.insert(&gd.acceleration_structures, AccelerationStructure {
            handle = build_info.dst,
            offset = u32(offset)
        }))
    }

    build_info.build_scratch_size = build_sizes.buildScratchSize

    // Queue build info for per-frame AS building step
    if create_info.type == .BOTTOM_LEVEL {
        queue.append(&gd.BLAS_queued_build_infos, build_info^)
    }

    return ret_handle
}

get_acceleration_structure :: proc(gd: ^Graphics_Device, handle: Acceleration_Structure_Handle) -> (^AccelerationStructure, bool) {
    as, b := hm.get(&gd.acceleration_structures, handle)
    return as, b
}

get_acceleration_structure_address :: proc(gd: ^Graphics_Device, handle: Acceleration_Structure_Handle) -> vk.DeviceAddress {
    blas, _ := hm.get(&gd.acceleration_structures, handle)
    addr_info := vk.AccelerationStructureDeviceAddressInfoKHR {
        sType = .ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        pNext = nil,
        accelerationStructure = blas.handle
    }
    return vk.GetAccelerationStructureDeviceAddressKHR(gd.device, &addr_info)
}

delete_acceleration_structure :: proc(gd: ^Graphics_Device, handle: Acceleration_Structure_Handle) -> bool {
    as := hm.get(&gd.acceleration_structures, handle) or_return
    queue.append(&gd.AS_deletes, AS_Delete {
        death_frame = gd.frame_count + u64(gd.frames_in_flight),
        handle = as.handle
    })
    hm.remove(&gd.acceleration_structures, handle)
    return true
}

cmd_build_queued_blases :: proc(gd: ^Graphics_Device) {
    if queue.len(gd.BLAS_queued_build_infos) > 0 {
        build_infos := make([dynamic]AccelerationStructureBuildInfo, 0, queue.len(gd.BLAS_queued_build_infos), context.temp_allocator)
        for queue.len(gd.BLAS_queued_build_infos) > 0 {
            as_build_info := queue.pop_front(&gd.BLAS_queued_build_infos)
            append(&build_infos, as_build_info)
        }
        cmd_build_acceleration_structures(gd, build_infos[:])
        gd.AS_required_scratch_size = 0
    }
}

cmd_build_acceleration_structures :: proc(
    gd: ^Graphics_Device,
    infos: []AccelerationStructureBuildInfo
) {
    cb_idx := CommandBuffer_Index(in_flight_idx(gd))
    cb := gd.gfx_command_buffers[in_flight_idx(gd)]     // @TODO: Use async compute queue instead

    // If scratch size is larger than current scratch buffer, reallocate scratch buffer
    if gd.AS_scratch_size < gd.AS_required_scratch_size {
        delete_buffer(gd, gd.AS_scratch_buffer)
    
        info := Buffer_Info {
            size = gd.AS_required_scratch_size,
            usage = {.STORAGE_BUFFER},
            required_flags = {.DEVICE_LOCAL},
            name = "Acceleration structure scratch buffer",
        }
        gd.AS_scratch_buffer = create_buffer(gd, &info)
        gd.AS_scratch_size = gd.AS_required_scratch_size

        log.warnf("Resized acceleration structure scratch buffer to %v bytes.", gd.AS_scratch_size)
    }

    scratch_buffer, ok := get_buffer(gd, gd.AS_scratch_buffer)
    if !ok {
        log.error("Couldn't get scratch buffer.")
    }

    g_infos := make([dynamic]vk.AccelerationStructureBuildGeometryInfoKHR, 0, len(infos), context.temp_allocator)
    range_infos := make([dynamic]vk.AccelerationStructureBuildRangeInfoKHR, 0, len(infos), context.temp_allocator)
    range_info_ptrs := make([dynamic][^]vk.AccelerationStructureBuildRangeInfoKHR, 0, len(infos), context.temp_allocator)
    scratch_addr_offset : vk.DeviceAddress = 0
    for info, i in infos {
        geos := make_geo_data_structs(gd, info.geometries[:])

        src, have_src := get_acceleration_structure(gd, info.src)
        src_handle : vk.AccelerationStructureKHR = 0
        if have_src && src.handle != 0 {
            src_handle = src.handle
        }

        build_info := vk.AccelerationStructureBuildGeometryInfoKHR {
            sType = .ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            pNext = nil,
            type = info.type,
            flags = info.flags,
            mode = info.mode,
            srcAccelerationStructure = src_handle,
            dstAccelerationStructure = info.dst,
            geometryCount = u32(len(geos)),
            pGeometries = raw_data(geos),
            ppGeometries = nil,
            scratchData = {
                deviceAddress = scratch_buffer.address + scratch_addr_offset
            }
        }
        range_info := vk.AccelerationStructureBuildRangeInfoKHR {
            primitiveCount = info.range_info.primitiveCount,
            primitiveOffset = info.range_info.primitiveOffset,
            firstVertex = info.range_info.firstVertex,
            transformOffset = info.range_info.transformOffset,
        }
        scratch_addr_offset += vk.DeviceAddress(info.build_scratch_size)
        append(&range_infos, range_info)
        append(&g_infos, build_info)
        append(&range_info_ptrs, &range_infos[i])
    }

    // Wait on any previous build commands' access to the
    // AS buffer and the scratch buffer
    as_buffer, _ := get_buffer(gd, gd.AS_buffer)
    {
        cmd_gfx_pipeline_barriers(gd, cb_idx, {
            {
                src_stage_mask = {.ACCELERATION_STRUCTURE_BUILD_KHR},
                src_access_mask = {.ACCELERATION_STRUCTURE_WRITE_KHR},
                dst_stage_mask = {.ACCELERATION_STRUCTURE_BUILD_KHR},
                dst_access_mask = {.ACCELERATION_STRUCTURE_WRITE_KHR},
                buffer = as_buffer.buffer,
                offset = 0,
                size = vk.DeviceSize(vk.WHOLE_SIZE),
            },
            {
                src_stage_mask = {.ACCELERATION_STRUCTURE_BUILD_KHR},
                src_access_mask = {.ACCELERATION_STRUCTURE_READ_KHR,.ACCELERATION_STRUCTURE_WRITE_KHR},
                dst_stage_mask = {.ACCELERATION_STRUCTURE_BUILD_KHR},
                dst_access_mask = {.ACCELERATION_STRUCTURE_READ_KHR,.ACCELERATION_STRUCTURE_WRITE_KHR},
                buffer = scratch_buffer.buffer,
                offset = 0,
                size = vk.DeviceSize(vk.WHOLE_SIZE),
            }
        }, {})
    }

    vk.CmdBuildAccelerationStructuresKHR(cb, u32(len(infos)), raw_data(g_infos), raw_data(range_info_ptrs))

    cmd_gfx_pipeline_barriers(gd, cb_idx, {
        {
            src_stage_mask = {.ACCELERATION_STRUCTURE_BUILD_KHR},
            src_access_mask = {.ACCELERATION_STRUCTURE_READ_KHR,.ACCELERATION_STRUCTURE_WRITE_KHR},
            dst_stage_mask = {.FRAGMENT_SHADER,.ACCELERATION_STRUCTURE_BUILD_KHR,.TRANSFER},
            dst_access_mask = {.ACCELERATION_STRUCTURE_READ_KHR,.TRANSFER_WRITE},
            buffer = as_buffer.buffer,
            offset = 0,
            size = vk.DeviceSize(vk.WHOLE_SIZE),
        }
    }, {})
}

