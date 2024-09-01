package desktop_vulkan_wrapper

import "core:fmt"
import "core:os"
import vk "vendor:vulkan"

import win32 "core:sys/windows"


Graphics_Device :: struct {
    // Basic Vulkan objects that every app definitely needs
    instance: vk.Instance,
    physical_device: vk.PhysicalDevice,
    device: vk.Device,
    pipeline_cache: vk.PipelineCache,
    

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

    
    window_support: bool        //Will this device need to draw to window surface swapchains?
}

vulkan_init :: proc(using feature_set: ^Init_Parameters) -> Graphics_Device {
    fmt.println("vk init go!")

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

        // Instead of forcing the caller to explicitly provide
        // the extensions they want to enable, I want to provide high-level
        // idioms that cover many extensions in the same logical category
        extensions: [dynamic]cstring

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
        
        if vk.CreateInstance(&create_info, nil, &inst) != vk.Result.SUCCESS {
            fmt.println("Instance creation failed.")
        }
    }

    // Load instance-level procedures
    vk.load_proc_addresses_instance(inst)

    // Create Vulkan device
    phys_device: vk.PhysicalDevice
    {

    }

    //Load proc addrs that come from the device driver
    //vk.load_proc_addresses_device()

    return Graphics_Device {}
}

