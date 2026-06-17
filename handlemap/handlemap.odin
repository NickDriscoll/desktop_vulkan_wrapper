// Wrapper around core Handle_Map type

package handlemap
import "base:builtin"
import hm "core:container/handle_map"

// Handle :: struct {
// 	generation: u32,
// 	index:      u32,
// }
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

// @(private)
// Sparse_Index :: struct {
// 	generation:    u32,
// 	index_or_next: u32,
// }

@(private)
Entity :: struct($T: typeid) {
	handle: Handle,
	data: T,
}


HandleMap :: struct($T: typeid) {
	// handles:        [dynamic]Handle,
	// values:         [dynamic]T,
	// sparse_indices: [dynamic]Sparse_Index,
	// next:           u32,
	handlemap: hm.Dynamic_Handle_Map(Entity(T), hm.Handle64)
}

init :: proc(m: ^$M/HandleMap($T), allocator := context.allocator) {
	// m^ = {}
	// m.handles.allocator        = allocator
	// m.values.allocator         = allocator
	// m.sparse_indices.allocator = allocator
	// m.next = 0
	hm.dynamic_init(&m.handlemap, allocator)
}

destroy :: proc(m: ^$M/HandleMap($T)) {
	// m^ = {}
	// clear(m)
	// delete(m.handles)
	// delete(m.values)
	// delete(m.sparse_indices)
	hm.dynamic_destroy(&m.handlemap)
}

clear :: proc(m: ^$M/HandleMap($T)) {
	// builtin.clear(&m.handles)
	// builtin.clear(&m.values)
	// builtin.clear(&m.sparse_indices)
	// m.next = 0
	hm.clear(&m.handlemap)
}

@(require_results)
has_handle :: proc(m: $M/HandleMap($T), h: $H/Handle) -> bool {
	// if h.index < u32(len(m.sparse_indices)) {
	// 	return m.sparse_indices[h.index].generation == h.generation
	// }
	// return false
	_, b := hm.dynamic_get(&m.handlemap, h)
	return b
}

@(require_results)
get :: proc(m: ^$M/HandleMap($T), h: $H/Handle) -> (^T, bool) {
	// if h.index < u32(len(m.sparse_indices)) {
	// 	entry := m.sparse_indices[h.index]
	// 	if entry.generation == h.generation {
	// 		return &m.values[entry.index_or_next], true
	// 	}
	// }
	// return nil, false
	return hm.get(&m.handlemap, h)
}

@(require_results)
insert :: proc(m: ^$M/HandleMap($T), value: T) -> (handle: Handle) {
	// if m.next < u32(len(m.sparse_indices)) {
	// 	entry := &m.sparse_indices[m.next]
	// 	assert(entry.generation < max(u32), "Generation sparse indices overflow")

	// 	entry.generation += 1
	// 	handle = Handle{
	// 		generation = entry.generation,
	// 		index = m.next,
	// 	}
	// 	m.next = entry.index_or_next
	// 	entry.index_or_next = u32(len(m.handles))
	// 	append(&m.handles, handle)
	// 	append(&m.values,  value)
	// } else {
	// 	assert(m.next < max(u32), "Index sparse indices overflow")

	// 	handle = Handle{
	// 		index = u32(len(m.sparse_indices)),
	// 	}
	// 	append(&m.sparse_indices, Sparse_Index{
	// 		index_or_next = u32(len(m.handles)),
	// 	})
	// 	append(&m.handles, handle)
	// 	append(&m.values,  value)
	// 	m.next += 1
	// }
	// return
	e := Entity(T) {
		data = value
	}
	h := hm.dynamic_add(&m.handlemap, e)

	return h
}

remove :: proc(m: ^$M/HandleMap($T), h: $H/Handle) -> bool {
	// if h.index < u32(len(m.sparse_indices)) {
	// 	entry := &m.sparse_indices[h.index]
	// 	if entry.generation != h.generation {
	// 		return
	// 	}
	// 	index := entry.index_or_next
	// 	entry.generation += 1
	// 	entry.index_or_next = m.next
	// 	m.next = h.index
	// 	value = m.values[index]
	// 	unordered_remove(&m.handles, int(index))
	// 	unordered_remove(&m.values,  int(index))
	// 	if index < u32(len(m.handles)) {
	// 		m.sparse_indices[m.handles[index].index].index_or_next = index
	// 	}
	// }
	// return
	return hm.remove(&m.handlemap, h)
}