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

// All buffers are read/written by the GPU using Buffer Device Address
// As such, we only need descriptor slots for images and samplers
Bindless_Descriptor_Bindings :: enum u32 {
    Images = 0,
    Samplers = 1
}

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

Sync_Info :: struct {
    wait_ops: [dynamic]Semaphore_Op,
    signal_ops: [dynamic]Semaphore_Op,
}

add_wait_op :: proc(gd: ^Graphics_Device, using i: ^Sync_Info, handle: Semaphore_Handle, value: u64) {
    append(&wait_ops, Semaphore_Op {
        semaphore = handle,
        value = value
    })
}

add_signal_op :: proc(gd: ^Graphics_Device, using i: ^Sync_Info, handle: Semaphore_Handle, value: u64) {
    append(&signal_ops, Semaphore_Op {
        semaphore = handle,
        value = value
    })
}

delete_sync_info :: proc(using s: ^Sync_Info) {
    delete(wait_ops)
    delete(signal_ops)
}

clear_sync_info :: proc(using s: ^Sync_Info) {
    clear(&wait_ops)
    clear(&signal_ops)
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
    physical_device_properties: vk.PhysicalDeviceProperties2,
    device: vk.Device,
    pipeline_cache: vk.PipelineCache,       // @TODO: Actually use pipeline cache
    alloc_callbacks: ^vk.AllocationCallbacks,
    allocator: vma.Allocator,
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


    // Pipeline layout used for all pipelines
    pipeline_layout: vk.PipelineLayout,
    
    // Handle_Maps of all Vulkan objects
    buffers: hm.Handle_Map(Buffer),
    images: hm.Handle_Map(Image),
    semaphores: hm.Handle_Map(vk.Semaphore),
    pipelines: hm.Handle_Map(vk.Pipeline),

    // Queue for images who's data has been uploaded
    // but which haven't been transferred to the gfx queue
    pending_images: queue.Queue(Pending_Image),

    // Deletion queues for Buffers and Images
    buffer_deletes: queue.Queue(Buffer_Delete),
    image_deletes: queue.Queue(Image_Delete),
    
}

API_Version :: enum {
    Vulkan12,
    Vulkan13
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
    api_version: API_Version,       // @TODO: Get rid of this?
    
    allocation_callbacks: ^vk.AllocationCallbacks,
    vk_get_instance_proc_addr: rawptr,
    
    frames_in_flight: u32,      // Maximum number of command buffers active at once
    
    
    window_support: bool        // Will this device need to draw to window surface swapchains?

}

init_vulkan :: proc(using params: ^Init_Parameters) -> Graphics_Device {
    assert(frames_in_flight > 0)
    
    log.log(.Info, "Initializing Vulkan instance and device")
    
    if vk_get_instance_proc_addr == nil {
        log.error("Init_Paramenters.vk_get_instance_proc_addr was nil!")
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

        append(&extensions, vk.EXT_DEBUG_UTILS_EXTENSION_NAME)
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
            pApplicationName = app_name,
            applicationVersion = app_version,
            pEngineName = engine_name,
            engineVersion = engine_version,
            apiVersion = api_version_int
        }
        create_info := vk.InstanceCreateInfo {
            sType = .INSTANCE_CREATE_INFO,
            pNext = &debug_info,
            flags = nil,
            pApplicationInfo = &app_info,
            enabledLayerCount = 0,
            ppEnabledLayerNames = nil,
            enabledExtensionCount = u32(len(extensions)),
            ppEnabledExtensionNames = raw_data(extensions)
        }
        
        if vk.CreateInstance(&create_info, allocation_callbacks, &inst) != .SUCCESS {
            log.error("Instance creation failed.")
        }
    }

    // Load instance-level procedures
    vk.load_proc_addresses_instance(inst)

    // Create Vulkan device
    phys_device: vk.PhysicalDevice
    properties: vk.PhysicalDeviceProperties2
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
            
            log.debugf("Considering physical device:\n%v", string_from_bytes(props.properties.deviceName[:]))

            // @TODO: Do something more sophisticated than picking the first DISCRETE_GPU
            if props.properties.deviceType == .DISCRETE_GPU {
                // Check physical device features
                vulkan_11_features: vk.PhysicalDeviceVulkan11Features
                vulkan_12_features: vk.PhysicalDeviceVulkan12Features
                dynamic_rendering_features: vk.PhysicalDeviceDynamicRenderingFeatures
                sync2_features: vk.PhysicalDeviceSynchronization2Features

                vulkan_11_features.sType = .PHYSICAL_DEVICE_VULKAN_1_1_FEATURES
                vulkan_12_features.sType = .PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
                dynamic_rendering_features.sType = .PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES
                sync2_features.sType = .PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES
                features.sType = .PHYSICAL_DEVICE_FEATURES_2

                vulkan_12_features.pNext = &vulkan_11_features
                dynamic_rendering_features.pNext = &vulkan_12_features
                sync2_features.pNext = &dynamic_rendering_features
                features.pNext = &sync2_features
                vk.GetPhysicalDeviceFeatures2(pd, &features)

                has_right_features :=
                    vulkan_11_features.variablePointers &&
                    vulkan_12_features.descriptorIndexing &&
                    vulkan_12_features.runtimeDescriptorArray &&
                    vulkan_12_features.timelineSemaphore &&
                    vulkan_12_features.bufferDeviceAddress && 
                    dynamic_rendering_features.dynamicRendering &&
                    sync2_features.synchronization2 
                if has_right_features {
                    phys_device = pd
                    properties = props
                    log.infof("Chosen GPU: %s", string_from_bytes(props.properties.deviceName[:]))
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
                    vk.EXT_MEMORY_BUDGET_EXTENSION_NAME
                }
            }
            case .Vulkan13: {
                necessary_extensions = {
                    vk.EXT_MEMORY_BUDGET_EXTENSION_NAME
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
                log.errorf("Your device does not support %v. Buh bye.", s)
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
        append(&extensions, vk.EXT_MEMORY_BUDGET_EXTENSION_NAME)
        if window_support {
            append(&extensions, vk.KHR_SWAPCHAIN_EXTENSION_NAME)
        }
        if api_version == .Vulkan12 {
            append(&extensions, vk.KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
            append(&extensions, vk.KHR_DYNAMIC_RENDERING_EXTENSION_NAME)
            append(&extensions, vk.KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)
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
            log.error("Failed to create device.")
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

        if vk.WaitSemaphores == nil do vk.WaitSemaphores = vk.WaitSemaphoresKHR
        if vk.WaitSemaphoresKHR == nil do vk.WaitSemaphoresKHR = vk.WaitSemaphores

        if vk.GetBufferDeviceAddress == nil do vk.GetBufferDeviceAddress = vk.GetBufferDeviceAddressKHR
        if vk.GetBufferDeviceAddressKHR == nil do vk.GetBufferDeviceAddressKHR = vk.GetBufferDeviceAddress
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
            flags = {.Externally_Synchronized,.Buffer_Device_Address,.Ext_Memory_Budget},
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
            log.error("Failed to initialize VMA.")
        }

        // Debug printing
        budget: vma.Budget
        vma.get_heap_budgets(allocator, &budget)
        log.debugf("%#v", budget)

        // stats: vma.Total_Statistics
        // vma.calculate_statistics(allocator, &stats)
        // log.debugf("%#v", stats)
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

        // Debug names
        assign_debug_name(device, .QUEUE, u64(uintptr(gfx_queue)), "GFX Queue")

        if compute_queue != gfx_queue {
            assign_debug_name(device, .QUEUE, u64(uintptr(compute_queue)), "Compute Queue")
        }

        if transfer_queue != gfx_queue {
            assign_debug_name(device, .QUEUE, u64(uintptr(transfer_queue)), "Transfer Queue")
        }
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
            log.error("Failed to create gfx command pool")
        }
        assign_debug_name(device, .COMMAND_POOL, u64(gfx_command_pool), "GFX Command Pool")
        
        compute_pool_info := vk.CommandPoolCreateInfo {
            sType = .COMMAND_POOL_CREATE_INFO,
            pNext = nil,
            flags = {vk.CommandPoolCreateFlag.TRANSIENT, vk.CommandPoolCreateFlag.RESET_COMMAND_BUFFER},
            queueFamilyIndex = compute_queue_family
        }
        if vk.CreateCommandPool(device, &compute_pool_info, allocation_callbacks, &compute_command_pool) != .SUCCESS {
            log.error("Failed to create compute command pool")
        }
        assign_debug_name(device, .COMMAND_POOL, u64(compute_command_pool), "Compute Command Pool")
        
        transfer_pool_info := vk.CommandPoolCreateInfo {
            sType = .COMMAND_POOL_CREATE_INFO,
            pNext = nil,
            flags = {vk.CommandPoolCreateFlag.TRANSIENT, vk.CommandPoolCreateFlag.RESET_COMMAND_BUFFER},
            queueFamilyIndex = transfer_queue_family
        }
        if vk.CreateCommandPool(device, &transfer_pool_info, allocation_callbacks, &transfer_command_pool) != .SUCCESS {
            log.error("Failed to create transfer command pool")
        }
        assign_debug_name(device, .COMMAND_POOL, u64(transfer_command_pool), "Transfer Command Pool")
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
            log.error("Failed to create gfx command buffers")
        }
        compute_info := vk.CommandBufferAllocateInfo {
            sType = .COMMAND_BUFFER_ALLOCATE_INFO,
            pNext = nil,
            commandPool = compute_command_pool,
            level = .PRIMARY,
            commandBufferCount = frames_in_flight
        }
        if vk.AllocateCommandBuffers(device, &compute_info, raw_data(compute_command_buffers)) != .SUCCESS {
            log.error("Failed to create compute command buffers")
        }
        transfer_info := vk.CommandBufferAllocateInfo {
            sType = .COMMAND_BUFFER_ALLOCATE_INFO,
            pNext = nil,
            commandPool = transfer_command_pool,
            level = .PRIMARY,
            commandBufferCount = frames_in_flight
        }
        if vk.AllocateCommandBuffers(device, &transfer_info, raw_data(transfer_command_buffers)) != .SUCCESS {
            log.error("Failed to create transfer command buffers")
        }

        // Debug naming
        sb: strings.Builder
        defer strings.builder_destroy(&sb)
        strings.builder_init(&sb, allocator = context.temp_allocator)
        for i in 0..<frames_in_flight {
            my_name := fmt.sbprintf(&sb, "GFX Command Buffer #%v", i)
            c_name := strings.unsafe_string_to_cstring(my_name)
            assign_debug_name(device, .COMMAND_BUFFER, u64(uintptr(gfx_command_buffers[i])), c_name)
            strings.builder_reset(&sb)
        }
        for i in 0..<frames_in_flight {
            my_name := fmt.sbprintf(&sb, "Compute Command Buffer #%v", i)
            c_name := strings.unsafe_string_to_cstring(my_name)
            assign_debug_name(device, .COMMAND_BUFFER, u64(uintptr(compute_command_buffers[i])), c_name)
            strings.builder_reset(&sb)
        }
        for i in 0..<frames_in_flight {
            my_name := fmt.sbprintf(&sb, "Transfer Command Buffer #%v", i)
            c_name := strings.unsafe_string_to_cstring(my_name)
            assign_debug_name(device, .COMMAND_BUFFER, u64(uintptr(transfer_command_buffers[i])), c_name)
            strings.builder_reset(&sb)
        }
    }

    // Create bindless descriptor set
    samplers: [TOTAL_IMMUTABLE_SAMPLERS]vk.Sampler
    ds_layout: vk.DescriptorSetLayout
    dp: vk.DescriptorPool
    ds: vk.DescriptorSet
    {
        // Create immutable samplers
        {
            sampler_infos := make([dynamic]vk.SamplerCreateInfo, len = 0, cap = TOTAL_IMMUTABLE_SAMPLERS, allocator = context.temp_allocator)
            defer delete(sampler_infos)
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

            debug_names : []cstring = {
                "Aniso 16 Sampler",
                "Point Sampler",
                "PostFX Sampler",
            }

            for &s, i in sampler_infos {
                vk.CreateSampler(device, &s, allocation_callbacks, &samplers[i])
                assign_debug_name(device, .SAMPLER, u64(samplers[i]), debug_names[i])
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
        bindings : []vk.DescriptorSetLayoutBinding = {image_binding, sampler_binding}

        binding_flags : vk.DescriptorBindingFlags = {.PARTIALLY_BOUND,.UPDATE_AFTER_BIND}
        binding_flags_plural : []vk.DescriptorBindingFlags = {binding_flags,binding_flags}
        binding_flags_info := vk.DescriptorSetLayoutBindingFlagsCreateInfo {
            sType = .DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            pNext = nil,
            bindingCount = 2,
            pBindingFlags = &binding_flags_plural[0]
        }
        layout_info := vk.DescriptorSetLayoutCreateInfo {
            sType = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext = &binding_flags_info,
            flags = {.UPDATE_AFTER_BIND_POOL},
            bindingCount = 2,
            pBindings = raw_data(bindings[:])
        }
        if vk.CreateDescriptorSetLayout(device, &layout_info, allocation_callbacks, &ds_layout) != .SUCCESS {
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
            log.error("Failed to create descriptor pool.")
        }

        set_info := vk.DescriptorSetAllocateInfo {
            sType = .DESCRIPTOR_SET_ALLOCATE_INFO,
            pNext = nil,
            descriptorPool = dp,
            descriptorSetCount = 1,
            pSetLayouts = &ds_layout
        }
        if vk.AllocateDescriptorSets(device, &set_info, &ds) != .SUCCESS {
            log.error("Failed to allocate descriptor sets.")
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
            log.error("Failed to create graphics pipeline layout.")
        }
    }

    gd := Graphics_Device {
        instance = inst,
        physical_device = phys_device,
        physical_device_properties = properties,
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

    // Create transfer timeline semaphore
    {
        info := Semaphore_Info {
            type = .TIMELINE,
            init_value = 0,
            name = "Transfer timeline"
        }
        gd.transfer_timeline = create_semaphore(&gd, &info)
    }

    return gd
}

quit_vulkan :: proc(gd: ^Graphics_Device) {
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

window_helper_plz_rename :: proc(gd: ^Graphics_Device, new_dims: hlsl.uint2) -> bool {
    // surface_caps := 
    // vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(gd.physical_device, gd.surface)

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
            width = new_dims.x,
            height = new_dims.y
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
        oldSwapchain = gd.swapchain
    }
    temp: vk.SwapchainKHR
    if vk.CreateSwapchainKHR(gd.device, &create_info, gd.alloc_callbacks, &temp) != .SUCCESS {
        return false
    }

    vk.DestroySwapchainKHR(gd.device, gd.swapchain, gd.alloc_callbacks)
    gd.swapchain = temp

    // Get swapchain images
    image_count : u32 = 0
    swapchain_images: [dynamic]vk.Image
    defer delete(swapchain_images)
    {
        vk.GetSwapchainImagesKHR(gd.device, gd.swapchain, &image_count, nil)
        resize(&swapchain_images, image_count)
        vk.GetSwapchainImagesKHR(gd.device, gd.swapchain, &image_count, raw_data(swapchain_images))
    }
    
    swapchain_image_views: [dynamic]vk.ImageView
    defer delete(swapchain_image_views)
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

    sb: strings.Builder
    defer strings.builder_destroy(&sb)
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
            gd.swapchain_images[i] = Image_Handle(hm.insert(&gd.images, im))

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

init_sdl2_window :: proc(gd: ^Graphics_Device, window: ^sdl2.Window) -> bool {
    if !sdl2.Vulkan_CreateSurface(window, gd.instance, &gd.surface) do return false

    width, height : i32 = 0, 0
    sdl2.Vulkan_GetDrawableSize(window, &width, &height)

    window_helper_plz_rename(gd, {u32(width), u32(height)}) or_return

    return true
}

resize_window :: proc(gd: ^Graphics_Device, new_dims: hlsl.uint2) -> bool {
    // The graphics device's swapchain should exist
    assert(gd.swapchain != 0)

    old_swapchain_handles := make([dynamic]Image_Handle, len(gd.swapchain_images), context.temp_allocator)
    defer delete(old_swapchain_handles)
    for handle, i in gd.swapchain_images do old_swapchain_handles[i] = handle

    window_helper_plz_rename(gd, new_dims) or_return

    // Remove stale swapchain image handles
    for handle in old_swapchain_handles do hm.remove(&gd.images, hm.Handle(handle))

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
        sharingMode = .CONCURRENT,          // @TODO: Actually evaluate if concurrent buffers on desktop are fine
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
    b.address = vk.GetBufferDeviceAddressKHR(gd.device, &bda_info)

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
    base_offset : u32 = 0
) -> bool {
    if len(in_slice) == 0 do return false

    cb := gd.transfer_command_buffers[in_flight_idx(gd)]
    out_buf := get_buffer(gd, out_buffer) or_return
    semaphore := hm.get(&gd.semaphores, hm.Handle(gd.transfer_timeline)) or_return
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
            if vk.QueueSubmit2KHR(gd.transfer_queue, 1, &submit_info, 0) != .SUCCESS {
                log.error("Failed to submit to transfer queue.")
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
    supports_mipmaps: bool,
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
    handle: Image_Handle,
    old_layout: vk.ImageLayout,
    new_layout: vk.ImageLayout,
    aspect_mask: vk.ImageAspectFlags,
    src_queue_family: u32
}

create_image :: proc(gd: ^Graphics_Device, using image_info: ^Image_Create) -> Image_Handle {
    // TRANSFER_DST is required for layout transitions period.
    // SAMPLED is required because all images in the system are
    // available in the bindless images array
    usage += {.SAMPLED,.TRANSFER_DST}

    // Calculate mipmap count
    mip_count : u32 = 1
    if supports_mipmaps {
        // Mipmaps are only for 2D images at least rn
        assert(image_type == .D2)

        highest_bit_32 :: proc(n: u32) -> u32 {
            n := n
            i : u32 = 0

            for n > 0 {
                n = n >> 1
                i += 1
            }
            return i
        }

        max_dim := max(extent.width, extent.height)
        mip_count = highest_bit_32(max_dim)
    }

    image: Image
    {
        info := vk.ImageCreateInfo {
            sType = .IMAGE_CREATE_INFO,
            pNext = nil,
            flags = flags,
            imageType = image_type,
            format = format,
            extent = extent,
            mipLevels = mip_count,
            arrayLayers = array_layers,
            samples = samples,
            tiling = tiling,
            usage = usage,
            sharingMode = .EXCLUSIVE,
            queueFamilyIndexCount = 1,
            pQueueFamilyIndices = &gd.gfx_queue_family,
            initialLayout = .UNDEFINED
        }
        alloc_info := vma.Allocation_Create_Info {
            flags = alloc_flags,
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
    image.extent = extent

    // Create the image view
    {
        view_type: vk.ImageViewType
        switch image_type {
            case .D1: view_type = .D1
            case .D2: view_type = .D2
            case .D3: view_type = .D3
        }

        aspect_mask : vk.ImageAspectFlags = {.COLOR}
        if format == .D32_SFLOAT {
            aspect_mask = {.DEPTH}
        }

        subresource_range := vk.ImageSubresourceRange {
            aspectMask = aspect_mask,
            baseMipLevel = 0,
            levelCount = u32(mip_count),
            baseArrayLayer = 0,
            layerCount = array_layers
        }

        info := vk.ImageViewCreateInfo {
            sType = .IMAGE_VIEW_CREATE_INFO,
            pNext = nil,
            flags = nil,
            image = image.image,
            viewType = view_type,
            format = format,
            components = IDENTITY_COMPONENT_SWIZZLE,
            subresourceRange = subresource_range
        }
        if vk.CreateImageView(gd.device, &info, gd.alloc_callbacks, &image.image_view) != .SUCCESS {
            log.error("Failed to create image view.")
        }
    }

    // Set debug names
    if len(name) > 0 {
        sb: strings.Builder
        defer strings.builder_destroy(&sb)
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

    return Image_Handle(hm.insert(&gd.images, image))
}

new_bindless_image :: proc(gd: ^Graphics_Device, using info: ^Image_Create, layout: vk.ImageLayout) -> Image_Handle {
    handle := create_image(gd, info)
    image, ok := hm.get(&gd.images, hm.Handle(handle))
    if !ok {
        log.error("Error in new_bindless_image()")
    }

    aspect_mask : vk.ImageAspectFlags = {.COLOR}
    if format == .D32_SFLOAT {
        aspect_mask = {.DEPTH}
    }

    queue.push_back(&gd.pending_images, Pending_Image {
        handle = handle,
        old_layout = .UNDEFINED,
        new_layout = layout,
        aspect_mask = aspect_mask,
        src_queue_family = gd.gfx_queue_family
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
    if !ok2 do log.error("Error getting transfer timeline semaphore")

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
                layerCount = 1
            }
        }
    }
    cmd_transfer_pipeline_barriers(gd, cb_idx, barriers)

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
    res := vk.QueueSubmit2KHR(gd.transfer_queue, 1, &submit_info, 0)
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

get_image :: proc(gd: ^Graphics_Device, handle: Image_Handle) -> (^Image, bool) {
    return hm.get(&gd.images, hm.Handle(handle))
}

get_image_vkhandle :: proc(gd: ^Graphics_Device, handle: Image_Handle) -> (h: vk.Image, ok: bool) {
    im := hm.get(&gd.images, hm.Handle(handle)) or_return
    return im.image, true
}

// Blocking function for creating a new GPU image with initial data
// Use when the data is small or if you don't care about stalling
sync_create_image_with_data :: proc(
    gd: ^Graphics_Device,
    using create_info: ^Image_Create,
    bytes: []byte
) -> (out_handle: Image_Handle, ok: bool) {
    // Create image first
    out_handle = create_image(gd, create_info)
    out_image := hm.get(&gd.images, hm.Handle(out_handle)) or_return

    using extent

    cb_idx := CommandBuffer_Index(in_flight_idx(gd))
    cb := gd.transfer_command_buffers[cb_idx]
    semaphore := hm.get(&gd.semaphores, hm.Handle(gd.transfer_timeline)) or_return

    // Staging buffer
    sb := hm.get(&gd.buffers, hm.Handle(gd.staging_buffer)) or_return
    sb_ptr := sb.alloc_info.mapped_data

    bytes_size := len(bytes)
    assert(bytes_size <= STAGING_BUFFER_SIZE, "Image too big for staging buffer. Nick, stop being lazy.")
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
                        aspectMask = {.COLOR},
                        baseMipLevel = 0,
                        levelCount = 1,
                        baseArrayLayer = 0,
                        layerCount = 1
                    }
                }
            }
            cmd_transfer_pipeline_barriers(gd, cb_idx, barriers)
        }

        image_copy := vk.BufferImageCopy {
            bufferOffset = vk.DeviceSize(amount_transferred),
            bufferRowLength = 0,
            bufferImageHeight = 0,
            imageSubresource = {
                aspectMask = {.COLOR},
                mipLevel = 0,
                baseArrayLayer = 0,
                layerCount = 1
            },
            imageOffset = {
                x = 0,
                y = 0,
                z = 0
            },
            imageExtent = {
                width = width,
                height = height,
                depth = depth
            }
        }
        vk.CmdCopyBufferToImage(cb, sb.buffer, out_image.image, .TRANSFER_DST_OPTIMAL, 1, &image_copy)

        // Record queue family ownership transfer on last iteration
        // @TODO: Barrier is overly opinionated about future usage
        if remaining_bytes <= STAGING_BUFFER_SIZE {
            barriers := []Image_Barrier {
                {
                    src_stage_mask = {.ALL_COMMANDS},
                    src_access_mask = {.MEMORY_READ,.MEMORY_WRITE},
                    dst_stage_mask = {.ALL_COMMANDS},                      // Ignored during release operation
                    dst_access_mask = {.MEMORY_READ,.MEMORY_WRITE},        // Ignored during release operation
                    old_layout = .TRANSFER_DST_OPTIMAL,
                    new_layout = .SHADER_READ_ONLY_OPTIMAL,
                    src_queue_family = gd.transfer_queue_family,
                    dst_queue_family = gd.gfx_queue_family,
                    image = out_image.image,
                    subresource_range = {
                        aspectMask = {.COLOR},
                        baseMipLevel = 0,
                        levelCount = 1,
                        baseArrayLayer = 0,
                        layerCount = 1
                    }
                }
            }
            cmd_transfer_pipeline_barriers(gd, cb_idx, barriers)
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
        res := vk.QueueSubmit2KHR(gd.transfer_queue, 1, &submit_info, 0)
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


    aspect_mask : vk.ImageAspectFlags = {.COLOR}
    if format == .D32_SFLOAT {
        aspect_mask = {.DEPTH}
    }

    pending_image := Pending_Image {
        handle = out_handle,
        old_layout = .TRANSFER_DST_OPTIMAL,
        new_layout = .SHADER_READ_ONLY_OPTIMAL,
        aspect_mask = aspect_mask,
        src_queue_family = gd.transfer_queue_family
    }
    queue.push_back(&gd.pending_images, pending_image)

    ok = true
    return
}

delete_image :: proc(gd: ^Graphics_Device, handle: Image_Handle) -> bool {
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

acquire_swapchain_image :: proc(gd: ^Graphics_Device, out_image_idx: ^u32) -> bool {
    idx := in_flight_idx(gd)
    sem := get_semaphore(gd, gd.acquire_semaphores[idx]) or_return
    
    if vk.AcquireNextImageKHR(gd.device, gd.swapchain, max(u64), sem^, 0, out_image_idx) != .SUCCESS {
        log.error("Failed to acquire swapchain image")
        return false
    }
    return true
}

present_swapchain_image :: proc(gd: ^Graphics_Device, image_idx: ^u32) -> bool {
    idx := in_flight_idx(gd)
    sem := get_semaphore(gd, gd.present_semaphores[idx]) or_return
    info_res: vk.Result
    info := vk.PresentInfoKHR {
        sType = .PRESENT_INFO_KHR,
        pNext = nil,
        waitSemaphoreCount = 1,
        pWaitSemaphores = sem,
        swapchainCount = 1,
        pSwapchains = &gd.swapchain,
        pImageIndices = image_idx,
        pResults = &info_res
    }
    if vk.QueuePresentKHR(gd.gfx_queue, &info) != .SUCCESS {
        log.error("Failed to present swapchain image.")
        return false
    }
    if info_res != .SUCCESS {
        log.error("Queue present individual failure.")
        return false
    }
    return true
}


CommandBuffer_Index :: distinct u32

begin_gfx_command_buffer :: proc(
    gd: ^Graphics_Device,
    sync_info: ^Sync_Info,
    gfx_timeline_semaphore: Semaphore_Handle
) -> CommandBuffer_Index {
    
    
    // Sync point where we wait if there are already N frames in the gfx queue
    if gd.frame_count >= u64(gd.frames_in_flight) {
        // Wait on timeline semaphore before starting command buffer execution
        wait_value := gd.frame_count - u64(gd.frames_in_flight) + 1
        // append(&sync_info.wait_ops, Semaphore_Op {
        //     semaphore = gfx_timeline_semaphore,
        //     value = wait_value
        // })
        
        // CPU-sync to prevent CPU from getting further ahead than
        // the number of frames in flight
        sem, ok := get_semaphore(gd, gfx_timeline_semaphore)
        if !ok do log.error("Couldn't find semaphore for CPU-sync")
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

    // Do per-frame that has to happen for any gfx command buffer
    {
        // Process buffer delete queue
        for queue.len(gd.buffer_deletes) > 0 && queue.peek_front(&gd.buffer_deletes).death_frame == gd.frame_count {
            buffer := queue.pop_front(&gd.buffer_deletes)
            log.debugf("Destroying buffer %s...", buffer.buffer)
            vma.destroy_buffer(gd.allocator, buffer.buffer, buffer.allocation)
        }

        // Process image delete queue
        for queue.len(gd.image_deletes) > 0 && queue.peek_front(&gd.image_deletes).death_frame == gd.frame_count {
            image := queue.pop_front(&gd.image_deletes)
            log.debugf("Destroying image %s...", image.image)
            vk.DestroyImageView(gd.device, image.image_view, gd.alloc_callbacks)
            vma.destroy_image(gd.allocator, image.image, image.allocation)
        }

        d_image_infos: [dynamic]vk.DescriptorImageInfo
        defer delete(d_image_infos)
        d_writes: [dynamic]vk.WriteDescriptorSet
        defer delete(d_writes)

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
                            levelCount = 1,
                            baseArrayLayer = 0,
                            layerCount = 1
                        }
                    }
                }
                cmd_gfx_pipeline_barriers(gd, cb_idx, barriers)

                // @TODO: Record mipmap generation commands

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
        if len(d_writes) > 0 do vk.UpdateDescriptorSets(gd.device, u32(len(d_writes)), &d_writes[0], 0, nil)
    }

    return cb_idx
}

submit_gfx_command_buffer :: proc(gd: ^Graphics_Device, cb_idx: CommandBuffer_Index, sync: ^Sync_Info) {
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
                stageMask = {.ALL_COMMANDS},    // @TODO: Is this heavy-handed or appropriate?
                deviceIndex = 0
            }
        }
        return true
    }

    // Make semaphore submit infos
    wait_submit_infos: [dynamic]vk.SemaphoreSubmitInfoKHR
    defer delete(wait_submit_infos)
    resize(&wait_submit_infos, len(sync.wait_ops))
    
    signal_submit_infos: [dynamic]vk.SemaphoreSubmitInfoKHR
    defer delete(signal_submit_infos)
    resize(&signal_submit_infos, len(sync.signal_ops))
    
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
        log.error("Unable to submit gfx command buffer")
    }
}













Framebuffer :: struct {
    color_images: [8]Image_Handle,
    depth_image: Image_Handle,
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


    vk.CmdBeginRenderingKHR(cb, &info)
}

cmd_bind_descriptor_set :: proc(gd: ^Graphics_Device, cb_idx: CommandBuffer_Index) {
    cb := gd.gfx_command_buffers[cb_idx]
    vk.CmdBindDescriptorSets(cb, .GRAPHICS, gd.pipeline_layout, 0, 1, &gd.descriptor_set, 0, nil)
}

cmd_bind_pipeline :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    bind_point: vk.PipelineBindPoint,
    handle: Pipeline_Handle
) -> bool {
    cb := gd.gfx_command_buffers[cb_idx]
    pipeline := hm.get(&gd.pipelines, hm.Handle(handle)) or_return
    vk.CmdBindPipeline(cb, bind_point, pipeline^)
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

cmd_push_constants_gfx :: proc($Struct_Type: typeid, gd: ^Graphics_Device, cb_idx: CommandBuffer_Index, in_struct: ^Struct_Type) {
    cb := gd.gfx_command_buffers[cb_idx]
    byte_count : u32 = u32(size_of(Struct_Type))
    vk.CmdPushConstants(cb, gd.pipeline_layout, {.VERTEX,.FRAGMENT}, 0, byte_count, in_struct)
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
    vk.CmdEndRenderingKHR(cb)
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
    image_barriers: []Image_Barrier
) {
    cb := gd.gfx_command_buffers[cb_idx]

    im_barriers: [dynamic]vk.ImageMemoryBarrier2
    defer delete(im_barriers)
    reserve(&im_barriers, len(image_barriers))
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
        bufferMemoryBarrierCount = 0,
        pBufferMemoryBarriers = nil,
        imageMemoryBarrierCount = u32(len(im_barriers)),
        pImageMemoryBarriers = raw_data(im_barriers)
    }
    vk.CmdPipelineBarrier2KHR(cb, &info)
}

// Inserts an arbitrary number of memory barriers
// into the transfer command buffer at this point
cmd_transfer_pipeline_barriers :: proc(
    gd: ^Graphics_Device,
    cb_idx: CommandBuffer_Index,
    image_barriers: []Image_Barrier
) {
    cb := gd.transfer_command_buffers[cb_idx]

    im_barriers: [dynamic]vk.ImageMemoryBarrier2
    defer delete(im_barriers)
    reserve(&im_barriers, len(image_barriers))
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
    if vk.CreateShaderModule(gd.device, &info, gd.alloc_callbacks, &mod) != .SUCCESS {
        log.error("Failed to create shader module.")
    }
    return mod
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
    defer delete(create_infos)
    resize(&create_infos, pipeline_count)

    pipelines: [dynamic]vk.Pipeline
    defer delete(pipelines)
    resize(&pipelines, pipeline_count)

    shader_modules: [dynamic]vk.ShaderModule
    defer delete(shader_modules)
    resize(&shader_modules, 2 * pipeline_count)

    defer for module in shader_modules {
        vk.DestroyShaderModule(gd.device, module, gd.alloc_callbacks)
    }    

    shader_infos: [dynamic]vk.PipelineShaderStageCreateInfo
    defer delete(shader_infos)
    resize(&shader_infos, 2 * pipeline_count)
    
    input_assembly_states: [dynamic]vk.PipelineInputAssemblyStateCreateInfo
    resize(&input_assembly_states, pipeline_count)
    defer delete(input_assembly_states)
    
    tessellation_states: [dynamic]vk.PipelineTessellationStateCreateInfo
    defer delete(tessellation_states)
    resize(&tessellation_states, pipeline_count)
    
    rasterization_states: [dynamic]vk.PipelineRasterizationStateCreateInfo
    defer delete(rasterization_states)
    resize(&rasterization_states, pipeline_count)
    
    multisample_states: [dynamic]vk.PipelineMultisampleStateCreateInfo
    defer delete(multisample_states)
    resize(&multisample_states, pipeline_count)
    
    sample_masks: [dynamic]^vk.SampleMask
    defer delete(sample_masks)
    resize(&sample_masks, pipeline_count)
    
    depthstencil_states: [dynamic]vk.PipelineDepthStencilStateCreateInfo
    defer delete(depthstencil_states)
    resize(&depthstencil_states, pipeline_count)

    colorblend_attachments: [dynamic]vk.PipelineColorBlendAttachmentState
    defer delete(colorblend_attachments)
    resize(&colorblend_attachments, pipeline_count)
    
    colorblend_states: [dynamic]vk.PipelineColorBlendStateCreateInfo
    defer delete(colorblend_states)
    resize(&colorblend_states, pipeline_count)
    
    renderpass_states: [dynamic]vk.PipelineRenderingCreateInfo
    defer delete(renderpass_states)
    resize(&renderpass_states, pipeline_count)
    
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
        shader_modules[2 * i] =     create_shader_module(gd, vertex_shader_bytecode)
        shader_modules[2 * i + 1] = create_shader_module(gd, fragment_shader_bytecode)
        shader_infos[2 * i] = vk.PipelineShaderStageCreateInfo {
            sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            stage = {.VERTEX},
            module = shader_modules[2 * i],
            pName = "main",
            pSpecializationInfo = nil
        }
        shader_infos[2 * i + 1] = vk.PipelineShaderStageCreateInfo {
            sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
            pNext = nil,
            flags = nil,
            stage = {.FRAGMENT},
            module = shader_modules[2 * i + 1],
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
            pSampleMask = sample_masks[i]
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
            blendConstants = cast([4]f32)colorblend_state.blend_constants
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
        log.error("Failed to compile graphics pipelines")
    }

    // Put newly created pipelines in the Handle_Map
    for p, i in pipelines {
        handles[i] = Pipeline_Handle(hm.insert(&gd.pipelines, p))
    }

    return handles
}