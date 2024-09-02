package desktop_vulkan_wrapper

import "core:fmt"
import "core:os"
import "core:log"
import vk "vendor:vulkan"

import win32 "core:sys/windows"

Graphics_Device :: struct {
    // Basic Vulkan objects that every app definitely needs
    instance: vk.Instance,
    physical_device: vk.PhysicalDevice,
    device: vk.Device,
    pipeline_cache: vk.PipelineCache,
    alloc_callbacks: ^vk.AllocationCallbacks,
    
    // The Vulkan queues that the device will submit on
    // May be aliases of each other if e.g. the GPU doesn't have
    // an async compute queue
    gfx_queue: vk.Queue,
    compute_queue: vk.Queue,
    transfer_queue: vk.Queue,

    // Command buffer related state
    gfx_command_pool: vk.CommandPool,
    compute_command_pool: vk.CommandPool,
    transfer_command_pool: vk.CommandPool,
    gfx_command_buffers: [dynamic]vk.CommandBuffer,
    compute_command_buffers: [dynamic]vk.CommandBuffer,
    transfer_command_buffers: [dynamic]vk.CommandBuffer

}

API_Version :: enum {
    Vulkan10,
    Vulkan11,
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

    frames_in_flight: u32,      // Maximum number of command buffers active at once

    
    window_support: bool        // Will this device need to draw to window surface swapchains?
}

vulkan_init :: proc(using params: ^Init_Parameters) -> Graphics_Device {
    assert(frames_in_flight > 0)

    log.log(.Info, "vk init go!")

    // @TODO: Support other OSes
    when ODIN_OS == .Windows {
        vk_dll := win32.LoadLibraryW(win32.utf8_to_wstring("vulkan-1.dll"))
        get_instance_proc_address := auto_cast win32.GetProcAddress(vk_dll, "vkGetInstanceProcAddr")
        vk.load_proc_addresses_global(get_instance_proc_address)
    }

    // Create Vulkan instance
    // @TODO: Look into vkEnumerateInstanceVersion()
    inst: vk.Instance
    {
        api_version_int: u32
        switch api_version {
            case .Vulkan10:
            case .Vulkan11:
                //Vulkan 1.0 is hot garbage
                panic("Minimum supported Vulkan is 1.2")
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
            
            if vk.QueueFlag.COMPUTE&vk.QueueFlag.TRANSFER in flags {
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
        
        // Create logical device
        create_info := vk.DeviceCreateInfo {
            sType = .DEVICE_CREATE_INFO,
            pNext = &features,
            flags = nil,
            queueCreateInfoCount = queue_count,
            pQueueCreateInfos = raw_data(&queue_create_infos),
            ppEnabledExtensionNames = nil,
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

    return Graphics_Device {
        instance = inst,
        physical_device = phys_device,
        device = device,

        alloc_callbacks = allocation_callbacks,
        gfx_queue = gfx_queue,
        compute_queue = compute_queue,
        transfer_queue = transfer_queue,
        gfx_command_pool = gfx_command_pool,
        compute_command_pool = compute_command_pool,
        transfer_command_pool = transfer_command_pool,
        gfx_command_buffers = gfx_command_buffers,
        compute_command_buffers = compute_command_buffers,
        transfer_command_buffers = transfer_command_buffers
    }
}

