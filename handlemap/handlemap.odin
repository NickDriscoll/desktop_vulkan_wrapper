// Wrapper around core Dynamic_Handle_Map type

package handlemap
import hm "core:container/handle_map"

Handle :: hm.Handle64

handle_to_u64 :: proc(h: $T/Handle) -> u64 {
	r: u64
	r |= u64(h.gen) << 32
	r |= u64(h.idx)
	return r
}

handle_to_rawptr :: proc(h: $T/Handle) -> rawptr {
	u: uintptr
	u |= (uintptr(h.gen) << 32)
	u |= (uintptr(h.idx) << 0)
	return rawptr(u)
}

u64_to_handle :: proc(r: u64) -> Handle {
	return Handle {
		gen = u32(r >> 32),
		idx = u32(r & 0xFFFFFFFF)
	}
}

rawptr_to_handle :: proc(r: rawptr) -> Handle {
	u := uintptr(r)

	return Handle {
		gen = u32(u >> 32),
		idx = u32(u & 0xFFFFFFFF)
	}
}

@(private)
Entity :: struct($T: typeid) {
	handle: Handle,
	data: T,
}


HandleMap :: struct($T: typeid) {
	handlemap: hm.Dynamic_Handle_Map(Entity(T), hm.Handle64)
}

init :: proc(m: ^$M/HandleMap($T), allocator := context.allocator) {
	hm.dynamic_init(&m.handlemap, allocator)
}

destroy :: proc(m: ^$M/HandleMap($T)) {
	hm.dynamic_destroy(&m.handlemap)
}

clear :: proc(m: ^$M/HandleMap($T)) {
	hm.clear(&m.handlemap)
}

len :: proc(m: $M/HandleMap($T)) -> uint {
	return hm.dynamic_len(m.handlemap)
}

@(require_results)
has_handle :: proc(m: $M/HandleMap($T), h: $H/Handle) -> bool {
	_, b := hm.dynamic_get(&m.handlemap, h)
	return b
}

@(require_results)
get :: proc(m: ^$M/HandleMap($T), h: $H/Handle) -> (^T, bool) {
	e, ok := hm.dynamic_get(&m.handlemap, Handle(h))
	return &e.data, ok
}

//HandleMapIterator :: hm.Dynamic_Handle_Map_Iterator(hm.Dynamic_Handle_Map(Entity($T), Handle))

@(require_results)
iterator_make :: proc(m: ^$M/HandleMap($T)) -> hm.Dynamic_Handle_Map_Iterator(hm.Dynamic_Handle_Map(Entity(T), Handle)) {
	return hm.dynamic_iterator_make(&m.handlemap)
}

@(require_results)
iterate :: proc(it: ^hm.Dynamic_Handle_Map_Iterator(hm.Dynamic_Handle_Map(Entity($T), Handle))) -> (^T, Handle, bool) {
	e, h, ok := hm.dynamic_iterate(it)
	return &e.data, h, ok
}

// Calls 'callback' for each element in handle_map
iterate_callback :: proc(handle_map: ^$M/HandleMap($T), userdata: rawptr, callback: proc(^T, Handle, rawptr)) {
	it := iterator_make(handle_map)
	for data, handle in iterate(&it) {
		callback(data, handle, userdata)
	}
}

@(require_results)
insert :: proc(m: ^$M/HandleMap($T), value: T) -> (handle: Handle) {
	e := Entity(T) {
		data = value
	}
	h := hm.dynamic_add(&m.handlemap, e)

	return h
}

remove :: proc(m: ^$M/HandleMap($T), h: $H/Handle) -> bool {
	ok, err := hm.dynamic_remove(&m.handlemap, Handle(h))
	assert(err == nil)
	return ok
}