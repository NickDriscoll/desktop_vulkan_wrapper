package desktop_vulkan_wrapper

import "core:fmt"
import "core:os"
import "core:log"
import "vendor:sdl2"
import vk "vendor:vulkan"

import win32 "core:sys/windows"

import hm "handlemap"

API_Version :: enum {
    Vulkan12,
    Vulkan13
}

Semaphore_Info :: struct {
    type: vk.SemaphoreType,
    init_value: u64,
}

Sync_Info :: struct {
    wait_values: [dynamic]u64,
    wait_semaphores: [dynamic]vk.Semaphore,
    signal_values: [dynamic]u64,
    signal_semaphores: [dynamic]vk.Semaphore,
}

Init_Parameters :: struct {
    // Vulkan instance creation parameters
    app_name: cstring,
    app_version: u32,
    engine_name: cstring,
    engine_version: u32,
    api_version: API_Version,
    
    allocation_callbacks: ^vk.AllocationCallbacks,
    
    frames_in_flight: u32,      // Maximum number of command buffers active at once
    
    
    window_support: bool        // Will this device need to draw to window surface swapchains?

}

// Distinct handle types for each Handle_Map in the Graphics_Device
VkSemaphore_Handle :: distinct hm.Handle

Graphics_Device :: struct {
    // Basic Vulkan objects that every app definitely needs
    instance: vk.Instance,
    physical_device: vk.PhysicalDevice,
    device: vk.Device,
    pipeline_cache: vk.PipelineCache,
    alloc_callbacks: ^vk.AllocationCallbacks,

    // Objects required to support windowing
    // Basically every app will use these, but maybe
    // these could be factored out
    surface: vk.SurfaceKHR,
    swapchain: vk.SwapchainKHR,

    frames_in_flight: u32,
    
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

    // Handle_Maps of all Vulkan objects
    semaphores: hm.Handle_Map(vk.Semaphore),
    pipelines: hm.Handle_Map(vk.Pipeline),


}

create_graphics_device :: proc(using params: ^Init_Parameters) -> Graphics_Device {
    assert(frames_in_flight > 0)

    log.log(.Info, "Initializing Vulkan instance; device")

    vk.load_proc_addresses_global(sdl2.Vulkan_GetVkGetInstanceProcAddr())

    // Create Vulkan instance
    // @TODO: Look into vkEnumerateInstanceVersion()
    inst: vk.Instance
    {
        api_version_int: u32
        switch api_version {
            case .Vulkan12:
                api_version_int = vk.API_VERSION_1_2
            case .Vulkan13:
                api_version_int = vk.API_VERSION_1_3
        }

        extensions: [dynamic]cstring
        defer delete(extensions)
        
        // Instead of forcing the caller to explicitly provide
        // the extensions they want to enable, I want to provide high-level
        // idioms that cover many extensions in the same logical category
        if window_support {
            append(&extensions, vk.KHR_SURFACE_EXTENSION_NAME)
            when ODIN_OS == .Windows {
                append(&extensions, vk.KHR_WIN32_SURFACE_EXTENSION_NAME)
            }
            when ODIN_OS == .Linux {
                VK_KHR_win32_surface
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
                features.sType = .PHYSICAL_DEVICE_FEATURES_2
                vk.GetPhysicalDeviceFeatures2(pd, &features)

                has_right_features := true
                if has_right_features {
                    phys_device = pd
                    break
                }
            }
        }

        assert(phys_device != nil, "Didn't find vkPhysicalDevice")
        log.debugf("%#v", features)

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

        comp_bytes_to_string :: proc(bytes: []byte, s: string) -> bool {
            return string(bytes[0:len(s)]) == s
        }

        // @TODO: Query for Sync2 support
        if api_version == .Vulkan12 {
            found_sync2 := false
            for ext in device_extensions {
                name := ext.extensionName
                if comp_bytes_to_string(name[:], vk.KHR_SYNCHRONIZATION_2_EXTENSION_NAME) {
                    found_sync2 = true
                    log.debug("Sync2 verified")
                    break
                }
            }
            if !found_sync2 {
                log.fatal("Your device does not support sync2. Buh bye.")
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
            append(&extensions, vk.KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
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

    gd := Graphics_Device {
        instance = inst,
        physical_device = phys_device,
        device = device,
        frames_in_flight = frames_in_flight,
        alloc_callbacks = allocation_callbacks,
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
    }

    // Init Handle_Maps
    {
        hm.init(&gd.semaphores)
        hm.init(&gd.pipelines)
    }

    return gd
}

init_sdl2_surface :: proc(vgd: ^Graphics_Device, window: ^sdl2.Window) -> bool {
    if !sdl2.Vulkan_CreateSurface(window, vgd.instance, &vgd.surface) do return false

    width, height : i32 = 0, 0
    sdl2.Vulkan_GetDrawableSize(window, &width, &height)

    // @TODO: Allow more configurability of swapchain options
    // particularly pertaining to presentation mode and image format
    create_info := vk.SwapchainCreateInfoKHR {
        sType = .SWAPCHAIN_CREATE_INFO_KHR,
        pNext = nil,
        flags = nil,
        surface = vgd.surface,
        minImageCount = vgd.frames_in_flight,
        imageFormat = .B8G8R8A8_SRGB,
        imageColorSpace = .SRGB_NONLINEAR,
        imageExtent = vk.Extent2D {
            width = u32(width),
            height = u32(height)
        },
        imageArrayLayers = 1,
        imageUsage = {.COLOR_ATTACHMENT},
        imageSharingMode = .EXCLUSIVE,
        queueFamilyIndexCount = 1,
        pQueueFamilyIndices = &vgd.gfx_queue_family,
        preTransform = {.IDENTITY},
        compositeAlpha = {.OPAQUE},
        presentMode = .FIFO,
        clipped = true,
        oldSwapchain = 0
    }
    swapchain: vk.SwapchainKHR
    if vk.CreateSwapchainKHR(vgd.device, &create_info, vgd.alloc_callbacks, &swapchain) != .SUCCESS {
        return false
    }

    return true
}

begin_gfx_command_buffer :: proc(gd: ^Graphics_Device) -> u32 {
    cb_idx := gd.next_gfx_command_buffer
    gd.next_gfx_command_buffer = (gd.next_gfx_command_buffer + 1) % gd.frames_in_flight

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

    return cb_idx
}

submit_gfx_command_buffer :: proc(gd: ^Graphics_Device, cb_idx: u32, sync: ^Sync_Info) {
    cb := gd.gfx_command_buffers[cb_idx]
    if vk.EndCommandBuffer(cb) != .SUCCESS {
        log.fatal("Unable to end gfx command buffer")
    }

    cb_info := vk.CommandBufferSubmitInfoKHR {
        sType = .COMMAND_BUFFER_SUBMIT_INFO_KHR,
        pNext = nil,
        commandBuffer = cb,
        deviceMask = 0
    }

    // Make semaphore submit infos
    wait_sem_count := len(sync.wait_semaphores)
    signal_sem_count := len(sync.signal_semaphores)
    wait_submit_infos: [dynamic]vk.SemaphoreSubmitInfoKHR
    signal_submit_infos: [dynamic]vk.SemaphoreSubmitInfoKHR
    defer delete(wait_submit_infos)
    defer delete(signal_submit_infos)
    resize(&wait_submit_infos, wait_sem_count)
    resize(&signal_submit_infos, signal_sem_count)
    for i := 0; i < wait_sem_count; i += 1 {
        wait_submit_infos[i] = vk.SemaphoreSubmitInfoKHR {
            sType = .SEMAPHORE_SUBMIT_INFO_KHR,
            pNext = nil,
            semaphore = sync.wait_semaphores[i],
            value = sync.wait_values[i],
            stageMask = nil,
            deviceIndex = 0
        }
    }
    for i := 0; i < signal_sem_count; i += 1 {
        signal_submit_infos[i] = vk.SemaphoreSubmitInfoKHR {
            sType = .SEMAPHORE_SUBMIT_INFO_KHR,
            pNext = nil,
            semaphore = sync.signal_semaphores[i],
            value = sync.signal_values[i],
            stageMask = nil,
            deviceIndex = 0
        }
    }

    info := vk.SubmitInfo2KHR {
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

create_semaphore :: proc(gd: ^Graphics_Device, info: ^Semaphore_Info) -> VkSemaphore_Handle {
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
    return VkSemaphore_Handle(hm.insert(&gd.semaphores, s))
}

check_timeline_semaphore :: proc(gd: ^Graphics_Device, handle: VkSemaphore_Handle) -> (val: u64, ok: bool) {
    sem := hm.get(&gd.semaphores, hm.Handle(handle)) or_return
    v: u64
    if vk.GetSemaphoreCounterValue(gd.device, sem^, &v) != .SUCCESS {
        log.fatal("Failed to read value from timeline semaphore")
        return 0, false
    }
    return v, true
}

get_semaphore :: proc(gd: ^Graphics_Device, handle: VkSemaphore_Handle) -> (^vk.Semaphore, bool) {
    return hm.get(&gd.semaphores, hm.Handle(handle))
}

destroy_semaphore :: proc(gd: ^Graphics_Device, handle: VkSemaphore_Handle) -> bool {
    semaphore := hm.get(&gd.semaphores, hm.Handle(handle)) or_return
    vk.DestroySemaphore(gd.device, semaphore^, gd.alloc_callbacks)
    
    return true
}