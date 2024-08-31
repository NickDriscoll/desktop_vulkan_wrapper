package desktop_vulkan_wrapper

import "core:fmt"
import vk "vendor:vulkan"

Graphics_Device :: struct {
    instance: vk.Instance,
    physical_device: vk.PhysicalDevice,
    device: vk.Device,
    pipeline_cache: vk.PipelineCache,
    

}

Graphics_FeatureSet :: struct {

}

vulkan_init :: proc(feature_set: Graphics_FeatureSet) {
    fmt.println("vk init go!")



    //vk.load_proc_addresses_instance()
}

